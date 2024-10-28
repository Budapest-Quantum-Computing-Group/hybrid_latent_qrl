from dataclasses import dataclass
from pathlib import Path
from collections import deque

from multiprocessing import Pipe
from tqdm import tqdm

from tf_function.ppoagent import PPOAgent
from tf_function.environment import Environment
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from datetime import datetime

import tf_function.ppoagent as ppoagent
import tf_function.critic as critic
import tf_function.latent_ae_policy as latent_ae_policy
import tf_function.latent_enc_policy as latent_enc_policy
import tf_function.latent_no_ae_training_policy as latent_no_ae_training_policy
import tf_function.latent_ae_policy_regularized as latent_ae_policy_regularized
import tf_function.policy as policy

import numpy as np

import tensorflow as tf
import piquasso as pq

import tyro
import json
import copy
import gc
import os
import shutil

from typing import List, Callable


@dataclass
class Args:
    config_file: Path
    save_path: Path
    agent_id: int
    continued: bool = False


VERBOSE = False


# Piquasso uses fp64, so we need to cast everything accordingly
# TODO these variables should be used in the other places too
dtype_tf = tf.float64
dtype_np = np.float64
tf.keras.backend.set_floatx('float64')


import logging


logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(pathname)s:%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)


def save_checkpoint(agent: PPOAgent, episode: int):
    config = agent.config
    logger.info(f"[Agent {agent.agent_id}] episode: {episode}/{config.episodes}, creating checkpoint...")

    ckpt_path = agent.save_path / "checkpoint"

    if ckpt_path.exists():
        shutil.rmtree(ckpt_path.as_posix())
    os.makedirs(ckpt_path.as_posix())

    with open(str(ckpt_path / "info.txt"), "w") as f:
        f.write(str({
            "episode": episode
        }))
        f.close()

    shutil.copyfile(
        str(agent.save_path / "progress.csv"),
        str(ckpt_path / "progress_checkpoint.csv")
    )

    for manager in agent.managers:
        manager.save()

    logger.info(f"[Agent {agent.agent_id}] episode: {episode}/{config.episodes}, checkpoint created.")


def run_multiprocesses(ppo_agent: PPOAgent):
    logger.info("Collecting data using run_multiprocesses...")

    works, parent_conns, child_conns = {}, {}, {}

    score_queue = deque(maxlen=ppo_agent.config.lr_last_run_count)

    for idx in range(ppo_agent.config.env_workers):
        parent_conn, child_conn = Pipe()

        work = Environment(
            env_idx = idx,
            child_conn = child_conn,
            config = ppo_agent.config,
            action_size = ppo_agent.env.action_space.n,
            visualize=False
        )

        work.start()

        logger.info(f"Started worker {idx}.")

        works[idx] = work
        parent_conns[idx] = parent_conn
        child_conns[idx] = child_conn

    states =        [[] for _ in range(ppo_agent.config.env_workers)]
    next_states =   [[] for _ in range(ppo_agent.config.env_workers)]
    actions =       [[] for _ in range(ppo_agent.config.env_workers)]
    rewards =       [[] for _ in range(ppo_agent.config.env_workers)]
    dones =         [[] for _ in range(ppo_agent.config.env_workers)]
    predictions =   [[] for _ in range(ppo_agent.config.env_workers)]
    scores =        [[] for _ in range(ppo_agent.config.env_workers)]
    score =         [0 for _ in range(ppo_agent.config.env_workers)]
    eplen =         [0 for _ in range(ppo_agent.config.env_workers)]

    state = [0 for _ in range(ppo_agent.config.env_workers)]

    logger.debug(f"list(parent_conns.keys()) = {list(parent_conns.keys())}")
    logger.debug(f"list(works.keys()) = {list(works.keys())}")
    logger.debug(f"list(child_conns.keys()) = {list(child_conns.keys())}")

    for worker_id in parent_conns.keys():
        state[worker_id] = parent_conns[worker_id].recv()
        logger.debug(f"state[worker_id].shape = {state[worker_id].shape}")
        logger.debug(f"state[worker_id].dtype = {state[worker_id].dtype}")

    episode = ppo_agent.episode_start
    decorator_classic = tf.function(jit_compile=True)
    enhanced_critic_call = decorator_classic(ppo_agent.critic.call)

    enhanced_encoder_call = None
    if ppo_agent.encoder is not None:
        enhanced_encoder_call = decorator_classic(ppo_agent.encoder.call)

    enhanced_decoder_call = None
    if ppo_agent.decoder is not None:
        enhanced_decoder_call = decorator_classic(ppo_agent.decoder.call)

    config = ppo_agent.config

    decorator_policy = tf.function(jit_compile=config.jit_compile)
    enhanced_policy_call = tf.function(jit_compile=config.jit_compile_call)(ppo_agent.policy.call)

    enhanced_opt_critic = decorator_classic(critic.optimize_critic)

    if config.use_latent_space_policy:
        if config.train_ae_policy:
            logger.info("Using latent_ae_policy module")
            if config.latent_width is None:
                enhanced_opt_policy = decorator_policy(latent_ae_policy.optimize_policy)
            else:
                enhanced_opt_policy = decorator_policy(latent_ae_policy_regularized.optimize_policy)
        elif config.train_enc_policy:
            logger.info("Using latent_enc_policy module")
            enhanced_opt_policy = decorator_policy(latent_enc_policy.optimize_policy)
        else:
            logger.info("Using latent_no_ae_training_policy module")
            enhanced_opt_policy = decorator_policy(latent_no_ae_training_policy.optimize_policy)
    else:
        logger.info("Using policy module")
        enhanced_opt_policy = decorator_policy(policy.optimize_policy)


    latent_width = None
    penalty_factor = None
    if config.latent_width is not None:
        latent_width = tf.convert_to_tensor(config.latent_width)
        penalty_factor = tf.convert_to_tensor(config.penalty_factor)

    while episode < ppo_agent.config.episodes:
        _state = np.reshape(state, [ppo_agent.config.env_workers, *(state[0].shape[1:])])

        __state = tf.convert_to_tensor(_state, dtype=dtype_tf)

        if ppo_agent.config.use_latent_space_policy:

            if ppo_agent.config.variational_ae:
                _, _, z = enhanced_encoder_call(
                    __state
                )

                predictions_list = enhanced_policy_call(
                    inputs=z
                ).numpy()
            else:
                z = enhanced_encoder_call(
                    __state
                )
                predictions_list = enhanced_policy_call(
                    inputs=z
                ).numpy()
        else:
            predictions_list = enhanced_policy_call(
                inputs=__state
            ).numpy()

        actions_list = [np.random.choice(ppo_agent.env.action_space.n, p=i) for i in predictions_list]

        for worker_id in parent_conns.keys():

            # Send action to connection
            parent_conns[worker_id].send(actions_list[worker_id])

            action_onehot = np.zeros([ppo_agent.env.action_space.n])
            action_onehot[actions_list[worker_id]] = 1
            actions[worker_id].append(action_onehot)
            predictions[worker_id].append(predictions_list[worker_id])

        for worker_id in parent_conns.keys():

            # Get rewards and next state
            next_state, reward, done, _ = parent_conns[worker_id].recv()

            states[worker_id].append(state[worker_id])
            next_states[worker_id].append(next_state)
            rewards[worker_id].append(reward)
            dones[worker_id].append(done)
            state[worker_id] = next_state
            score[worker_id] += reward
            eplen[worker_id] += 1

            # done includes truncation
            if done:
                logger.info(f"[Agent {ppo_agent.agent_id}] episode: {episode}/{ppo_agent.config.episodes}, worker: {worker_id}, eplen: {eplen[worker_id]}, score: {score[worker_id]}|{datetime.now()}")

                with open(ppo_agent.save_path / "progress.csv", 'a') as f:
                    f.write(f"{episode}|{worker_id}|{eplen[worker_id]}|{score[worker_id]}|{datetime.now()}\n")
                    f.close()

                scores[worker_id].append(score[worker_id])

                score_queue.append(score[worker_id])

                score[worker_id] = 0
                eplen[worker_id] = 0
                if(episode < ppo_agent.config.episodes):
                    episode += 1

                if(episode > 0 and episode % ppo_agent.config.save_freq == 0):
                    save_checkpoint(agent=ppo_agent, episode=episode)


        for worker_id in works.keys():
            if len(states[worker_id]) < ppo_agent.config.replay_batch:
                continue

            # set learning rate based on the mean of the replay batches
            if config.is_reward_based_lr_scheduler and len(score_queue) == score_queue.maxlen:
                mean_scores = np.mean(np.array(score_queue))
                mean_transformed_score = ppo_agent.transform_rewards(mean_scores)
                # ppo_agent.policy_lr_scheduler should not be None here
                lrnow = ppo_agent.policy_lr_scheduler(score=mean_transformed_score, episode=episode)
                logger.info(f"Setting lr to {lrnow} [scores: {score_queue}, mean scores: {mean_scores}]")
                ppo_agent.policy_optimizer.learning_rate.assign(lrnow)

            logger.info(f"[Agent {ppo_agent.agent_id}] [worker {worker_id}] Replaying...")
            replay(
                states=states[worker_id],
                actions=actions[worker_id],
                rewards=rewards[worker_id],
                predictions=predictions[worker_id],
                dones=dones[worker_id],
                next_states=next_states[worker_id],
                encoder=ppo_agent.encoder,
                decoder=ppo_agent.decoder,
                critic=ppo_agent.critic,
                policy=ppo_agent.policy,
                enhanced_encoder_call=enhanced_encoder_call,
                enhanced_decoder_call=enhanced_decoder_call,
                enhanced_critic_call=enhanced_critic_call,
                opt_critic=enhanced_opt_critic,
                opt_policy=enhanced_opt_policy,
                latent_width=latent_width,
                penalty_factor=penalty_factor
            )

            states[worker_id] = []
            next_states[worker_id] = []
            actions[worker_id] = []
            rewards[worker_id] = []
            dones[worker_id] = []
            predictions[worker_id] = []
            scores[worker_id] = []
            logger.info(f"[Agent {ppo_agent.agent_id}] [worker {worker_id}] Collecting data...")

    # terminating processes after while loop
    for key in works.keys():
        works[key].terminate()
        logger.info(f'PROCESS TERMINATED: {works[key]}')
        works[key].join()


def get_gaes(rewards, dones, values, next_values, discount_factor = 0.99, gae_lambda = 0.9, normalize=True):
    deltas = [r + discount_factor * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * discount_factor * gae_lambda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return np.vstack(gaes), np.vstack(target)


def create_dataset(data):
    logger.info(f"data: {data}")
    dataset = tf.data.Dataset.from_tensor_slices({
        'states': data[:, 0],
        'next_states': data[:, 1],
        'actions': data[:, 2],
        'predictions': data[:, 3],
        'values': data[:, 4],
        'next_values': data[:, 5],
        'advantages': data[:, 6],
        'target': data[:, 7]
    })
    return dataset.batch(ppo_agent.config.training_minibatch)


def replay(
    states: List[np.ndarray],
    actions: List[np.ndarray],
    rewards: List[np.ndarray],
    predictions: List[np.ndarray],
    dones: List[np.ndarray],
    next_states: List[np.ndarray],
    encoder: Model,
    decoder: Model,
    critic: Model,
    policy: Model,
    enhanced_encoder_call: Model,
    enhanced_decoder_call: Model,
    enhanced_critic_call: Model,
    opt_critic: Callable,
    opt_policy: Callable,
    latent_width: tf.Tensor,
    penalty_factor: tf.Tensor
):
    states = np.vstack(states)
    next_states = np.vstack(next_states)
    actions = np.vstack(actions)
    predictions = np.stack(predictions)
    rewards = np.vstack(rewards)
    dones = np.vstack(dones)

    logger.info(f"states.dtype={states.dtype}")

    if ppo_agent.config.use_latent_space_critic:
        if ppo_agent.config.variational_ae:
                _, _, z = enhanced_encoder_call(states)
                _, _, next_z = enhanced_encoder_call(next_states)
                values      = enhanced_critic_call(z)
                next_values = enhanced_critic_call(next_z)
        else:
            z = enhanced_encoder_call(states)
            next_z = enhanced_encoder_call(next_states)
            values      = enhanced_critic_call(z)
            next_values = enhanced_critic_call(next_z)

    else:
        values      = enhanced_critic_call(states)
        next_values = enhanced_critic_call(next_states)

    advantages, target = get_gaes(
        rewards, dones, np.squeeze(values), np.squeeze(next_values),
        discount_factor = ppo_agent.config.discount_factor,
        gae_lambda = ppo_agent.config.gae_lambda,
        normalize = ppo_agent.config.normalize_gae
    )

    if VERBOSE:
        logger.info(f"states = {states}")
        logger.info(f"type(states)={type(states)}")
        logger.info(f"states.shape = {states.shape}")

        logger.info(f"next_states.shape = {next_states.shape}")
        logger.info(f"actions.shape = {actions.shape}")
        logger.info(f"rewards.shape = {rewards.shape}")
        logger.info(f"dones.shape = {dones.shape}")
        logger.info(f"predictions.shape = {predictions.shape}")
        logger.info(f"values.shape = {values.shape}")
        logger.info(f"next_values.shape = {next_values.shape}")
        logger.info(f"advantages.shape = {advantages.shape}")
        logger.info(f"target.shape = {target.shape}")

    dataset = tf.data.Dataset.from_tensor_slices({
        'states': states,
        'next_states': next_states,
        'actions': actions,
        'predictions': predictions,
        'values': values,
        'next_values': next_values,
        'advantages': advantages,
        'target': target
    }).batch(ppo_agent.config.training_minibatch)

    config = ppo_agent.config

    for ee in range(ppo_agent.config.training_epochs):
        logger.info(f"[Agent {ppo_agent.agent_id}] replay epoch {ee}...")
        for batch in tqdm(dataset):
            states = batch['states']
            next_states = batch['next_states']
            actions = batch['actions']
            predictions = batch['predictions']
            values = batch['values']
            next_values = batch['next_values']
            advantages = batch['advantages']
            target = batch['target']

            opt_critic(
                encoder=encoder,
                decoder=decoder,
                critic=critic,
                opt=ppo_agent.critic_optimizer,
                loss_clipping=tf.convert_to_tensor(ppo_agent.config.critic_loss_clipping, dtype=dtype_tf),
                states=states,
                next_states=next_states,
                actions=actions,
                predictions=predictions,
                values=values,
                next_values=next_values,
                advantages=advantages,
                target=target,
                latent_width=latent_width,
                penalty_factor=penalty_factor
            )

            opt_policy(
                encoder=encoder,
                decoder=decoder,
                policy=policy,
                opt=ppo_agent.policy_optimizer,
                states=states,
                next_states=next_states,
                actions=actions,
                predictions=predictions,
                values=values,
                next_values=next_values,
                advantages=advantages,
                target=target,
                epsilon_clip=tf.convert_to_tensor(config.epsilon_clip, dtype=dtype_tf),
                entropy_coeff=tf.convert_to_tensor(config.entropy_coeff, dtype=dtype_tf),
                ae_coeff_policy=tf.convert_to_tensor(config.ae_coeff_policy, dtype=dtype_tf),
                latent_width=latent_width,
                penalty_factor=penalty_factor
            )

            gc.collect()

        gc.collect()
    gc.collect()
    logger.info("Replay done")


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU Memory allocation set to dynamic.")
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"Number of physical GPUs: {len(gpus)}")
        logger.info(f"Number of physical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.critical(e)


if __name__ == "__main__":
    args = tyro.cli(Args)

    tf.get_logger().setLevel("ERROR")

    logger.info(f"Loading config file from {args.config_file}...")
    with open(args.config_file, "r") as f:
        ppo_config = ppoagent.Config(**json.loads(f.read()))
    logger.info(f"Configs loaded.")

    ppo_agent = ppoagent.new_agent(ppo_config, args.save_path,
                                   args.agent_id, args.continued)
    run_multiprocesses(ppo_agent)
