from dataclasses import dataclass
from pathlib import Path

import tf_function.transform_rewards

from gymnasium import Env
from contextlib import redirect_stdout
import numpy as np

import gymnasium as gym
import sys

import classical_critics

import tensorflow as tf
import shutil
import ast
import importlib

import piquasso as pq
import os

from tensorflow.keras import optimizers
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

import latent_qrl_envs

from typing import Optional, Callable, List

import logging


logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

def mbexpdim(x):
    if len(x.shape)==1:
        return tf.expand_dims(
            x, axis=0, name=None
        )
    else:
        return x


@dataclass
class Config:
    env_name: str
    env_kwargs: dict
    env_wrappers: list
    env_workers: int

    use_latent_space_critic: bool
    use_latent_space_policy: bool

    train_enc_critic: bool
    train_enc_policy: bool

    train_ae_critic: bool
    train_ae_policy: bool

    variational_ae: bool

    ae_coeff_critic: float
    ae_coeff_policy: float

    beta_vae: float
    beta_scheduler_name: str
    beta_scheduler_kwargs: dict

    encoder_classname: str
    encoder_kwargs: dict

    decoder_classname: str
    decoder_kwargs: dict

    policy_module: str
    policy_name: str
    policy_kwargs: dict

    critic_name: str
    critic_kwargs: dict

    critic_optimizer_name: str
    critic_lr: float
    # critic_lr_scheduler_name: str
    # critic_lr_scheduler_kwargs: dict

    policy_optimizer_name: str
    policy_lr: float

    policy_lr_scheduler_module: str
    policy_lr_scheduler_name: Optional[str]
    policy_lr_scheduler_kwargs: dict
    lr_last_run_count: int
    is_reward_based_lr_scheduler: float

    transform_rewards_name: str
    transform_rewards_kwargs: dict

    epsilon_clip: float
    entropy_coeff: float
    gae_lambda: float
    discount_factor: float
    critic_loss_clipping: float
    normalize_gae: bool

    episodes: int
    max_episode_len: int
    training_epochs: int
    training_minibatch: int
    replay_batch: int
    save_freq: int
    jit_compile: bool
    jit_compile_call: bool

    profile: bool

    latent_width: Optional[float] = None
    penalty_factor: float = 10.0

    state_preprocessing: Optional[str] = None
    pretrained_encoder: Optional[Path] = None
    pretrained_decoder: Optional[Path] = None



@dataclass
class PPOAgent:
    config: Config

    save_path: Path
    agent_id: int
    episode_start: int

    env: Env

    critic_optimizer: OptimizerV2
    policy_optimizer: OptimizerV2

    replay_count: int
    state_preprocessing: Callable

    policy_lr_scheduler: Callable

    transform_rewards: Callable

    policy: Model
    critic: Model

    managers: List[tf.train.CheckpointManager]

    encoder: Optional[Model]
    decoder: Optional[Model]


def new_agent(config: Config, save_path: Path,
              agent_id: int, continued: bool) -> PPOAgent:

    episode_start = 0
    save_path = save_path / str(agent_id)
    if not continued:
        os.makedirs(save_path)


    from gymnasium.spaces import Box, Discrete
    env = gym.make(config.env_name, **config.env_kwargs)

    import env_wrappers

    for wrapper_params in config.env_wrappers:
            logger.info(f"Wrapping environment with wrapper {wrapper_params['name']}...")
            env = getattr(env_wrappers, wrapper_params['name'])(env, **wrapper_params['kwargs'])

    assert type(env.observation_space) is Box, f"type of env.observation_space must be Box, got {type(env.observation_space)}"
    assert type(env.action_space) is Discrete, f"type of env.action_space must be Discrete, got {type(env.action_space)}"

    if (config.train_enc_critic == True or config.train_ae_critic == True) and config.use_latent_space_critic == False:
        logger.critical(f"Incompatible settings.")

    if (config.train_enc_policy == True or config.train_ae_policy == True) and config.use_latent_space_policy == False:
        logger.critical(f"Incompatible settings.")

    if config.train_ae_critic == True:
        logger.info(f"train_ae_critic is True, ignoring train_enc_critic.")

    if config.train_ae_policy == True:
        logger.info(f"train_ae_policy is True, ignoring train_enc_policy.")

    if config.policy_lr_scheduler_name is not None:
        m = importlib.import_module(config.policy_lr_scheduler_module)
        policy_lr_scheduler = getattr(m, config.policy_lr_scheduler_name)(initial_lr=config.policy_lr,
                                                                          **config.policy_lr_scheduler_kwargs)
    else:
        policy_lr_scheduler = None

    if config.is_reward_based_lr_scheduler:
        learning_rate = config.policy_lr
    else:
        if config.policy_lr_scheduler_name is not None:
            learning_rate = policy_lr_scheduler
        else:
            learning_rate = config.policy_lr

    transform_rewards = getattr(tf_function.transform_rewards, config.transform_rewards_name)(**config.transform_rewards_kwargs)

    critic_optimizer = getattr(optimizers, config.critic_optimizer_name)(learning_rate=config.critic_lr)
    policy_optimizer = getattr(optimizers, config.policy_optimizer_name)(learning_rate=learning_rate)

    replay_count = 0

    import state_preprocessings
    if config.state_preprocessing:
        try:
            state_preprocessing = getattr(state_preprocessings, config.state_preprocessing)
            logger.info(f"Using state_preprocessing: {config.state_preprocessing}.")
        except Exception as e:
            logger.critical(f"Couldn't instantiate state_preprocessing: {e}")
            sys.exit(-1)
    else:
        state_preprocessing = lambda x:x

    m = importlib.import_module(config.policy_module)
    decorator = tf.function(jit_compile=config.jit_compile)
    policy = getattr(m, config.policy_name)(**config.policy_kwargs, decorator=decorator)
    logger.info(f"Using {config.policy_name}.")

    critic = getattr(classical_critics, config.critic_name)(**config.critic_kwargs)

    # Call networks on dummy data

    s0, _ = env.reset()
    logger.info(f"s0: {s0.shape}")
    s0 = state_preprocessing(s0)

    encoder = None
    decoder = None

    ckpt_path = save_path / "checkpoint"

    managers = [
        tf.train.CheckpointManager(
            tf.train.Checkpoint(critic=critic),
            str(ckpt_path / "critic"),
            max_to_keep=1,
            checkpoint_name="critic"
        ),
        tf.train.CheckpointManager(
            tf.train.Checkpoint(critic_optimizer=critic_optimizer),
            str(ckpt_path / "critic_optimizer"),
            max_to_keep=1,
            checkpoint_name="critic_optimizer"
        ),
        tf.train.CheckpointManager(
            tf.train.Checkpoint(policy=policy),
            str(ckpt_path / "policy"),
            max_to_keep=1,
            checkpoint_name="policy"
        ),
        tf.train.CheckpointManager(
            tf.train.Checkpoint(policy_optimizer=policy_optimizer),
            str(ckpt_path / "policy_optimizer"),
            max_to_keep=1,
            checkpoint_name="policy_optimizer"
        )
    ]

    if config.use_latent_space_critic or config.use_latent_space_policy:
        # Trick to load saved fp32 weights and convert the model back to fp64.
        # May need to be rewritten for better code readability.

        def get_loaded_weights(saved_dtype = 'float32', needed_dtype = 'float64'):

            if tf.keras.backend.floatx() == needed_dtype:

                tf.keras.backend.set_floatx(saved_dtype)

                import ae_encoders, ae_decoders
                dummy_encoder = getattr(ae_encoders, config.encoder_classname)(**config.encoder_kwargs)
                dummy_decoder = getattr(ae_decoders, config.decoder_classname)(**config.decoder_kwargs)

                if config.variational_ae:
                    _, _, z0 = dummy_encoder(tf.cast([s0], saved_dtype))
                else:
                    z0 = dummy_encoder(tf.cast([s0], saved_dtype))
                s0rec  = dummy_decoder(z0)

                if config.pretrained_encoder:
                    try:
                        logger.info(f"Loading weights for encoder from path {config.pretrained_encoder}")
                        dummy_encoder.load_weights(config.pretrained_encoder)
                    except Exception as e:
                        logger.critical(f"Couldn't load encoder weights from {config.pretrained_encoder}\n")
                        logger.critical(e)
                        sys.exit(-1)

                if config.pretrained_decoder:
                    try:
                        logger.info(f"Loading weights for decoder from path {config.pretrained_decoder}")
                        dummy_decoder.load_weights(config.pretrained_decoder)
                    except Exception as e:
                        logger.critical(f"Couldn't load decoder weights from {config.pretrained_decoder}\n")
                        sys.exit(-1)

                encoder_weights = []
                for w in dummy_encoder.get_weights():
                    encoder_weights.append(w.astype(needed_dtype))

                decoder_weights = []
                for w in dummy_decoder.get_weights():
                    decoder_weights.append(w.astype(needed_dtype))

                tf.keras.backend.set_floatx(needed_dtype)

                return encoder_weights, decoder_weights


        import ae_encoders, ae_decoders
        encoder = getattr(ae_encoders, config.encoder_classname)(**config.encoder_kwargs)
        decoder = getattr(ae_decoders, config.decoder_classname)(**config.decoder_kwargs)

        managers.extend([
            tf.train.CheckpointManager(
                tf.train.Checkpoint(encoder=encoder),
                str(ckpt_path / "encoder"),
                max_to_keep=1,
                checkpoint_name="encoder"
            ),
            tf.train.CheckpointManager(
                tf.train.Checkpoint(decoder=decoder),
                str(ckpt_path / "decoder"),
                max_to_keep=1,
                checkpoint_name="decoder"
            )
        ])

        # Initialize encoder and decoder weights
        if config.variational_ae:
            _, _, z0 = encoder(tf.cast([s0], 'float64'))
        else:
            z0 = encoder(tf.cast([s0], 'float64'))
        s0rec  = decoder(z0)

        # ew and dw should contain the saved weights casted to fp64
        ew, dw = get_loaded_weights()

        encoder.set_weights(ew)
        decoder.set_weights(dw)

        def assert_all_fp64(model):
            for layer in model.layers:
                    for i, weight in enumerate(layer.weights):
                        assert weight.dtype == tf.float64, f"Found fp32 parameter when fp64 is expected: {weight}"

        assert_all_fp64(encoder)
        assert_all_fp64(decoder)

    # weights = tf.Variable(pq.cvqnn.generate_random_cvqnn_weights(layer_count=config.layer_count,
    #                                                              d=config.modes), dtype=tf.float64)

    # weights_checkpoint = tf.train.Checkpoint(weights=weights)

    critic.build((None, *critic.input_shape_))

    # TODO make it more readable
    if config.use_latent_space_policy:
        x = policy(inputs=z0)
        logger.info(f"x={x}")
        critic_optimizer.build(list(encoder.trainable_variables) +
                            list(decoder.trainable_variables) +
                            list(critic.trainable_variables))
        policy_optimizer.build(list(encoder.trainable_variables) +
                            list(decoder.trainable_variables) +
                            list(policy.trainable_variables))
    else:
        _ = policy(
            inputs=tf.expand_dims(
                s0, axis=0, name=None
            )
        )

        critic_optimizer.build(list(critic.trainable_variables))
        policy_optimizer.build(list(policy.trainable_variables))

    # Print summaries if not continued.
    if not continued:
        if encoder is not None:
            with open(save_path / "encoder_summary.txt", 'w') as f:
                with redirect_stdout(f):
                    encoder.summary()
                f.close()

        if decoder is not None:
            with open(save_path / "decoder_summary.txt", 'w') as f:
                with redirect_stdout(f):
                    decoder.summary()
                f.close()

        with open(save_path / "critic_summary.txt", 'w') as f:
            with redirect_stdout(f):
                critic.summary()
            f.close()

        with open(save_path / "policy_summary.txt", 'w') as f:
            with redirect_stdout(f):
                policy.summary()
            f.close()

        with open(save_path / f"progress.csv", 'w') as f:
            f.write("episode|worker_id|eplen|score|ts\n")
            f.close()

    else:
        # Try loading checkpoint info.
        with open(str(ckpt_path / "info.txt"), "r") as f:
            ckpt_ep = ast.literal_eval(f.read())["episode"]
            episode_start = ckpt_ep + 1
            f.close()
            logger.info(f"[Agent {agent_id}]: Continuing from episode {ckpt_ep}.")

        logger.info(f"[Agent {agent_id}]: Restoring progress file...")
        shutil.copyfile(
            str(ckpt_path / "progress_checkpoint.csv"),
            str(save_path / "progress.csv")
        )

        for manager in managers:
            manager.restore_or_initialize()
        logger.info("Models are loaded")

    logger.info("Agent initialized.")


    return PPOAgent(
        config=config,
        save_path=save_path,
        agent_id=agent_id,
        episode_start=episode_start,
        env=env,
        critic_optimizer=critic_optimizer,
        policy_optimizer=policy_optimizer,
        replay_count=replay_count,
        state_preprocessing=state_preprocessing,
        policy=policy,
        critic=critic,
        encoder=encoder,
        decoder=decoder,
        transform_rewards=transform_rewards,
        policy_lr_scheduler=policy_lr_scheduler,
        managers=managers
    )
