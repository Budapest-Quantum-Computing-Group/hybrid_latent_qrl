import os

import shutil
import sys
import ast
import gc
import random
import gymnasium as gym
import pylab
import numpy as np
import copy

from tqdm import tqdm

# def tqdm(x):
#     return x

from threading import Thread, Lock
from multiprocessing import Process, Pipe
import multiprocessing 
multiprocessing.set_start_method('spawn', force=True)  # this line is needed for tf and multiprocessing to work properly

import time

from datetime import datetime

from inspect import getmembers, isfunction

from contextlib import redirect_stdout

import tensorflow as tf
global tf  # this line is needed for tf and multiprocessing to work properly

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config-file', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--agent-id', type=int, required=True)
parser.add_argument('--continued', action='store_true')

args = parser.parse_args()

# Piquasso uses fp64, so we need to cast everything accordingly
dtype_tf = tf.float64
dtype_np = np.float64
tf.keras.backend.set_floatx('float64')

import logging

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

# USAGE:
# logger.debug("This is a debug log")
# logger.info("This is an info log")
# logger.critical("This is critical")
# logger.error("An error occurred")

VERBOSE = False

tf.get_logger().setLevel("ERROR")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("GPU Memory allocation set to dynamic.")
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"Number of physical GPUs: {len(gpus)}")
        logger.info(f"Number of physical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.critical(e)

from JSONConfig import JSONConfig

config = JSONConfig()

config.add_key('env_name', type=str, default="CarRacing-v2")
config.add_key('env_kwargs', type=dict)
config.add_key('env_wrappers', type=list)
config.add_key('env_workers', type=int, default = 4)

config.add_key('use_latent_space_critic', type=bool, required=True)
config.add_key('use_latent_space_policy', type=bool, required=True)

config.add_key('train_enc_critic', type=bool, required=True)
config.add_key('train_enc_policy', type=bool, required=True)

config.add_key('train_ae_critic', type=bool, required=True)
config.add_key('train_ae_policy', type=bool, required=True)

config.add_key('variational_ae', type=bool, default= False)
        
config.add_key('ae_coeff_critic', type=float, default = 0.99)
config.add_key('ae_coeff_policy', type=float, default = 0.99)

config.add_key('beta_vae', type=float, default = 0.05)
config.add_key('beta_scheduler_name', type=str)
config.add_key('beta_scheduler_kwargs', type=dict)

config.add_key('state_preprocessing', type=str)

config.add_key('encoder_classname', type=str)
config.add_key('encoder_kwargs', type=dict)
config.add_key('pretrained_encoder', type=str)

config.add_key('decoder_classname', type=str)
config.add_key('decoder_kwargs', type=dict)
config.add_key('pretrained_decoder', type=str)

config.add_key('policy_name', type=str)
config.add_key('policy_kwargs', type=dict)

config.add_key('critic_name', type=str)
config.add_key('critic_kwargs', type=dict)

config.add_key('critic_optimizer_name', type=str, default='Adam')
config.add_key('critic_lr', type=float, default=0.00025)
config.add_key('critic_lr_scheduler_name', type=str)
config.add_key('critic_lr_scheduler_kwargs', type=dict)

config.add_key('policy_optimizer_name', type=str, default='Adam')
config.add_key('policy_lr', type=float, default=0.00025)
config.add_key('policy_lr_scheduler_name', type=str)
config.add_key('policy_lr_scheduler_kwargs', type=dict)

config.add_key('epsilon_clip', type=float, required=True, default=0.2)
config.add_key('entropy_coeff', type=float, required=True, default=0.2)
config.add_key('gae_lambda', type=float, required=True, default=0.2)
config.add_key('discount_factor', type=float, required=True, default=0.2)
config.add_key('critic_loss_clipping', type=float, required=True, default=0.2)
config.add_key('normalize_gae', type=bool, required=True, default=True)

config.add_key('episodes', type=int, default = 1000_000)
config.add_key('max_episode_len', type=int, required=True)
config.add_key('training_epochs', type=int, default=10)
config.add_key('training_minibatch', type=int, default=32)
config.add_key('replay_batch', type=int, default = 1000)
config.add_key('save_freq', type=int, required=True, default=500)

config.add_key('profile', type=bool, required=False, default=False)

logger.info(f"Loading config file from {args.config_file}...")
config.load(args.config_file)
logger.info(f"Configs loaded.")

tf.get_logger().setLevel("ERROR")

def mbexpdim(x):
    if len(x.shape)==1:
        return tf.expand_dims(
            x, axis=0, name=None
        )
    else:
        return x


class Environment(Process):
    def __init__(self, env_idx, child_conn, config, action_size, visualize=False):
        super(Environment, self).__init__()
        
        self.is_render = visualize
        self.maxeplen = config.max_episode_len
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.config = config
        # self.state_size = state_size

        self.env = gym.make(config.env_name, **config.env_kwargs)
        self.env.action_space.n = action_size

        self.current_step = 0

        import env_wrappers

        for wrapper_params in config.env_wrappers:
            logger.info(f"[Environment {self.env_idx}]: Wrapping environment with wrapper {wrapper_params['name']}...")
            self.env = getattr(env_wrappers, wrapper_params['name'])(self.env, **wrapper_params['kwargs'])

        import state_preprocessings
        if self.config.state_preprocessing:
            try: 
                self.state_preprocessing = getattr(state_preprocessings, self.config.state_preprocessing)
                logger.info(f"[Environment {self.env_idx}]: Using state_preprocessing: {self.config.state_preprocessing}.")
            except Exception as e:
                logger.critical(f"[Environment {self.env_idx}]: Couldn't instantiate state_preprocessing: {e}")
                sys.exit(-1)
        else:
            self.state_preprocessing = self.state_preprocessing = getattr(state_preprocessings, 'identity')

        logger.info(f"[Environment {self.env_idx}]: initialized.")

    def run(self):
        logger.info(f"[Environment {self.env_idx}]: running...")
        super(Environment, self).run()

        logger.info(f"[Environment {self.env_idx}]: resetting...")
        state, _ = self.env.reset()
        state = self.state_preprocessing(state)
        logger.info(f"[Environment {self.env_idx}]: is reset.")
        logger.info(f"[Environment {self.env_idx}]: max_episode_len = {self.config.max_episode_len}")

        state = np.reshape(state, [1, *state.shape])
        
        self.child_conn.send(state)
        
        while True:
            action = self.child_conn.recv()
            
            if self.is_render and self.env_idx == 0:
                self.env.render()

            state, reward, done, info, _ = self.env.step(action)
            state = self.state_preprocessing(state)
            state = np.reshape(state, [1, *state.shape])
                
            if self.current_step >= self.config.max_episode_len:
                done = True
                
            if done:
                self.current_step = 0
                state, _ = self.env.reset()
                state = self.state_preprocessing(state)
                state = np.reshape(state, [1, *state.shape])
            
            self.current_step = self.current_step + 1
            self.child_conn.send([state, reward, done, info])


class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, config, save_path, agent_id, continued):
        # Initialization
        # Environment and PPO parameters
        self.config=config
        self.save_path = save_path
        self.agent_id = agent_id
        self.continued = continued
        self.episode_start = 0

        from gymnasium.spaces import Box, Discrete
        self.env = gym.make(self.config.env_name, **config.env_kwargs)
        
        import env_wrappers

        for wrapper_params in config.env_wrappers:
            logger.info(f"Wrapping environment with wrapper {wrapper_params['name']}...")
            self.env = getattr(env_wrappers, wrapper_params['name'])(self.env, **wrapper_params['kwargs'])

        assert type(self.env.observation_space) is Box, f"type of env.observation_space must be Box, got {type(self.env.observation_space)}"
        assert type(self.env.action_space) is Discrete, f"type of env.action_space must be Discrete, got {type(self.env.action_space)}"

        if (self.config.train_enc_critic == True or self.config.train_ae_critic == True) and self.config.use_latent_space_critic == False:
            logger.critical(f"Incompatible settings.")
        
        if (self.config.train_enc_policy == True or self.config.train_ae_policy == True) and self.config.use_latent_space_policy == False:
            logger.critical(f"Incompatible settings.")

        if self.config.train_ae_critic == True:
            logger.info(f"train_ae_critic is True, ignoring train_enc_critic.")
        
        if self.config.train_ae_policy == True:
            logger.info(f"train_ae_policy is True, ignoring train_enc_policy.")

    
        self.critic_optimizer = getattr(optimizers, self.config.critic_optimizer_name)(learning_rate=self.config.critic_lr)
        self.policy_optimizer = getattr(optimizers, self.config.policy_optimizer_name)(learning_rate=self.config.policy_lr)

        # TODO: Add lr schedulers
        
        self.replay_count = 0

        import state_preprocessings
        if self.config.state_preprocessing:
            try: 
                self.state_preprocessing = getattr(state_preprocessings, self.config.state_preprocessing)
                logger.info(f"Using state_preprocessing: {self.config.state_preprocessing}.")
            except Exception as e:
                logger.critical(f"Couldn't instantiate state_preprocessing: {e}")
                sys.exit(-1)
        else:
            self.state_preprocessing = lambda x:x


        

        import classical_policies, qnn_policies, pqcvqnn
        
        try:
            logger.info(f"Trying to instantiate classical policy...")
            self.policy = getattr(classical_policies, self.config.policy_name)(**self.config.policy_kwargs)
            logger.info(f"Using {self.config.policy_name}.")
        except Exception as e:
            logger.error(f"Error while trying to load {self.config.policy_name} from classical_policies:\nERROR: {e}")
            try: 
                logger.info(f"Trying to instantiate QNN policy...")
                self.policy = getattr(qnn_policies, self.config.policy_name)(**self.config.policy_kwargs)
                logger.info(f"Using {self.config.policy_name}.")
            except Exception as e:
                logger.error(f"Error while trying to load {self.config.policy_name} from qnn_policies:\nERROR: {e}")
                try: 
                    logger.info(f"Trying to instantiate PQCVQNN policy...")
                    self.policy = getattr(pqcvqnn.policies, self.config.policy_name)(**self.config.policy_kwargs)
                    logger.info(f"Using {self.config.policy_name}.")
                except Exception as e:
                    logger.error(f"Error while trying to load {self.config.policy_name} from pqcvqnn.policies:\nERROR: {e}")
                    try: 
                        logger.info(f"Trying to instantiate Batched PQCVQNN policy...")
                        self.policy = getattr(pqcvqnn.policies_batched, self.config.policy_name)(**self.config.policy_kwargs)
                        logger.info(f"Using {self.config.policy_name}.")
                    except Exception as e:
                        logger.critical(f"Couldn't instantiate any policy: {e}")
                        sys.exit(-1)

        import classical_critics
        self.critic = getattr(classical_critics, self.config.critic_name)(**self.config.critic_kwargs)

        # Call networks on dummy data

        s0, _ = self.env.reset()
        s0 = self.state_preprocessing(s0)

        self.encoder = None
        self.decoder = None
        

        if self.config.use_latent_space_critic or self.config.use_latent_space_policy:

            # Trick to load saved fp32 weights and convert the model back to fp64.
            # May need to be rewritten for better code readability.

            def get_loaded_weights(saved_dtype = 'float32', needed_dtype = 'float64'):

                if tf.keras.backend.floatx() == needed_dtype:
                    
                    tf.keras.backend.set_floatx(saved_dtype)
                    
                    import ae_encoders, ae_decoders
                    dummy_encoder = getattr(ae_encoders, self.config.encoder_classname)(**self.config.encoder_kwargs)
                    dummy_decoder = getattr(ae_decoders, self.config.decoder_classname)(**self.config.decoder_kwargs)

                    if self.config.variational_ae:
                        _, _, z0 = dummy_encoder(tf.cast([s0], saved_dtype))
                    else:
                        z0 = dummy_encoder(tf.cast([s0], saved_dtype))
                    s0rec  = dummy_decoder(z0)

                    if self.config.pretrained_encoder:
                        try: 
                            logger.info(f"Loading weights for encoder from path {self.config.pretrained_encoder}")
                            dummy_encoder.load_weights(self.config.pretrained_encoder)
                        except Exception as e:
                            logger.critical(f"Couldn't load encoder weights from {self.config.pretrained_encoder}\n")
                            logger.critical(e)
                            sys.exit(-1)
                    
                    if self.config.pretrained_decoder:
                        try: 
                            logger.info(f"Loading weights for decoder from path {self.config.pretrained_decoder}")
                            dummy_decoder.load_weights(self.config.pretrained_decoder)
                        except Exception as e:
                            logger.critical(f"Couldn't load decoder weights from {self.config.pretrained_decoder}\n")
                            logger.critical(e)
                            sys.exit(-1)

                    encoder_weights = []
                    for w in dummy_encoder.get_weights():
                        encoder_weights.append(w.astype(needed_dtype))

                    decoder_weights = []
                    for w in dummy_decoder.get_weights():
                        decoder_weights.append(w.astype(needed_dtype))

                    del dummy_encoder, dummy_decoder

                    tf.keras.backend.set_floatx(needed_dtype)

                    return encoder_weights, decoder_weights
            

            import ae_encoders, ae_decoders
            self.encoder = getattr(ae_encoders, self.config.encoder_classname)(**self.config.encoder_kwargs)
            self.decoder = getattr(ae_decoders, self.config.decoder_classname)(**self.config.decoder_kwargs)

            # Initialize encoder and decoder weights
            if self.config.variational_ae:
                _, _, z0 = self.encoder(tf.cast([s0], 'float64'))
            else:
                z0 = self.encoder(tf.cast([s0], 'float64'))
            s0rec  = self.decoder(z0)
           
            # ew and dw should contain the saved weights casted to fp64
            ew, dw = get_loaded_weights()

            self.encoder.set_weights(ew)
            self.decoder.set_weights(dw)

            def assert_all_fp64(model):
                for layer in model.layers:
                        for i, weight in enumerate(layer.weights):
                            assert weight.dtype == tf.float64, f"Found fp32 parameter when fp64 is expected: {weight}"
                            
            assert_all_fp64(self.encoder)
            assert_all_fp64(self.decoder)

        
        if self.config.use_latent_space_critic:
            _  = self.critic(z0)
        else:
            _ = self.critic(mbexpdim(s0))

        if self.config.use_latent_space_policy:
            _ = self.policy(z0)
        else:
            _ = self.policy(mbexpdim(s0))

        # Print summaries if not continued.
        if not self.continued:
            if self.encoder: 
                with open(os.path.join(self.save_path, "encoder_summary.txt"), 'w') as f:
                    with redirect_stdout(f):
                        self.encoder.summary()
                    f.close()

            if self.decoder: 
                with open(os.path.join(self.save_path, "decoder_summary.txt"), 'w') as f:
                    with redirect_stdout(f):
                        self.decoder.summary()
                    f.close()

            with open(os.path.join(self.save_path, "critic_summary.txt"), 'w') as f:
                with redirect_stdout(f):
                    self.critic.summary()
                f.close()

            with open(os.path.join(self.save_path, "policy_summary.txt"), 'w') as f:
                with redirect_stdout(f):
                    self.policy.summary()
                f.close()

            with open(os.path.join(self.save_path, f"progress.csv"), 'w') as f:
                f.write("episode|worker_id|eplen|score|ts\n")
                f.close()
        
        else: 
            # Try loading checkpoint info.
            ckpt_path = os.path.join(self.save_path, "checkpoint")
            
            with open(os.path.join(ckpt_path, "info.txt"), "r") as f:
                ckpt_ep = ast.literal_eval(f.read())["episode"]
                self.episode_start = ckpt_ep
                f.close()
                logger.info(f"[Agent {self.agent_id}]: Continuing from episode {ckpt_ep}.")

            logger.info(f"[Agent {self.agent_id}]: Restoring progress file...")
            
            shutil.copyfile(
                os.path.join(ckpt_path, "progress_checkpoint.csv"),
                os.path.join(self.save_path, "progress.csv")
            )

            # Save models
            if self.encoder: 
                self.encoder.load_weights(os.path.join(ckpt_path, "encoder"))
                logger.info(f"[Agent {self.agent_id}]: Encoder weights loaded from checkpoint.")

            if self.decoder: 
                self.decoder.load_weights(os.path.join(ckpt_path, "decoder"))
                logger.info(f"[Agent {self.agent_id}]: Decoder weights loaded from checkpoint.")

            self.critic.load_weights(os.path.join(ckpt_path, "critic"))
            logger.info(f"[Agent {self.agent_id}]: Critic weights loaded from checkpoint.")
            
            self.policy.load_weights(os.path.join(ckpt_path, "policy"))
            logger.info(f"[Agent {self.agent_id}]: Policy weights loaded from checkpoint.")
        
        logger.info("Agent initilaized.")

    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        
        # Use the network to predict the next action to take, using the model
        
        prediction = self.policy(state) # [0]
        
        action = np.random.choice(self.env.action_space.n, p=prediction)
        
        action_onehot = np.zeros([self.env.action_space.n])
        
        action_onehot[action] = 1
        
        # prediction is the probs
        return action, action_onehot, prediction

    def discount_rewards(self, reward, discount_factor):#gaes is better
        # Compute the discount_factor-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * discount_factor + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, discount_factor = 0.99, gae_lambda = 0.9, normalize=True):
        deltas = [r + discount_factor * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * discount_factor * gae_lambda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

        
    def ppo_loss(self, advantages, prediction_picks, actions, predicted_probs, epsilon_clip, entropy_coeff):
        # Defined in https://arxiv.org/abs/1707.06347

        if VERBOSE: 
            logger.debug(f"advantages.shape = {advantages.shape}")
            logger.debug(f"prediction_picks.shape = {prediction_picks.shape}")
            logger.debug(f"actions.shape = {actions.shape}")
            logger.debug(f"predicted_probs.shape = {predicted_probs.shape}")
        
        prob     = actions * predicted_probs
        old_prob = actions * prediction_picks

        if VERBOSE: 
            logger.debug(f"prob.shape = {prob.shape}")
            logger.debug(f"old_prob.shape = {old_prob.shape}")


        prob     = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - epsilon_clip, max_value=1 + epsilon_clip) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(predicted_probs * K.log(predicted_probs + 1e-10))
        entropy = entropy_coeff * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def critic_loss(self, y_true, y_pred, values, loss_clipping):
        if VERBOSE: 
            logger.debug(f"y_true.shape = {y_true.shape}")
            logger.debug(f"y_pred.shape = {y_pred.shape}")
            logger.debug(f"values.shape = {values.shape}")

        clipped_value_loss = values + K.clip(y_pred - values, -loss_clipping, loss_clipping)

        v_loss1 = (y_true - clipped_value_loss) ** 2
        v_loss2 = (y_true - y_pred) ** 2
        
        value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))    

        return value_loss

    def critic_loss_simple(self, targets,  values):
        value_loss = K.mean((targets - values) ** 2) # standard PPO loss
        return value_loss

    def optimize_critic(self, states, next_states, actions, predictions, values, next_values, advantages, target):
        with tf.GradientTape() as tape:

            if self.config.use_latent_space_critic:
                
                if self.config.variational_ae:
                    z_mean, z_log_var, z = self.encoder( states )
                    y_pred = self.critic(z)
                else:
                    z = self.encoder( states )
                    y_pred = self.critic(z)
            
            else:
                y_pred = self.critic(states)

            c_loss = self.critic_loss(
                target, y_pred, values,
                loss_clipping = self.config.critic_loss_clipping
            )

            total_loss_critic = c_loss

            if self.config.use_latent_space_critic and self.config.train_ae_critic:
                
                s_rec = self.decoder( z )
                ae_loss = tf.math.reduce_mean((s_rec - states)**2)

                if self.config.variational_ae:
                    kl_loss = tf.math.reduce_mean(
                        -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    )

                    ae_loss = ae_loss + self.config.beta_vae * kl_loss

                total_loss_critic = total_loss_critic + self.config.ae_coeff_critic * ae_loss
                
        
        if self.config.use_latent_space_critic:
            
            if self.config.train_ae_critic:
                critic_grads = tape.gradient(
                    total_loss_critic,
                    self.critic.trainable_variables + self.encoder.trainable_variables + self.decoder.trainable_variables
                )

                self.critic_optimizer.apply_gradients(zip(
                    critic_grads, 
                    self.critic.trainable_variables + self.encoder.trainable_variables + self.decoder.trainable_variables
                ))

            # Not training decoder, but training encoder
            elif self.config.train_enc_critic:
                critic_grads = tape.gradient(
                    total_loss_critic,
                    self.critic.trainable_variables + self.encoder.trainable_variables
                )

                self.critic_optimizer.apply_gradients(zip(
                    critic_grads, 
                    self.critic.trainable_variables + self.encoder.trainable_variables
                ))

        # Not training AE at all, only critic
        else:
            critic_grads = tape.gradient(
                total_loss_critic,
                self.critic.trainable_variables
            )

            self.critic_optimizer.apply_gradients(zip(
                critic_grads, 
                self.critic.trainable_variables
            ))

        # free memory
        del total_loss_critic, tape

    def optimize_policy(self, states, next_states, actions, predictions, values, next_values, advantages, target):
        with tf.GradientTape() as tape:

            if self.config.use_latent_space_policy:
                if self.config.variational_ae:
                    z_mean, z_log_var, z = self.encoder( states )
                    predicted_probs = self.policy(z)
                else:
                    z = self.encoder( states )
                    predicted_probs = self.policy(z)
                    
            else:
                predicted_probs = self.policy(states)
            

            a_loss = self.ppo_loss(
                advantages,
                predictions,
                actions,
                predicted_probs,
                epsilon_clip = self.config.epsilon_clip,
                entropy_coeff = self.config.entropy_coeff
            )

            if VERBOSE: 
                logger.debug(f"a_loss.dtype={a_loss.dtype}")
                logger.debug(f"a_loss.shape={a_loss.shape}")
                logger.debug(f"a_loss={a_loss}" )

            total_loss_policy = a_loss

            if self.config.use_latent_space_policy and self.config.train_ae_policy:
                s_rec = self.decoder( z )
                ae_loss = tf.math.reduce_mean((s_rec - states)**2)

                if self.config.variational_ae:
                    kl_loss = tf.math.reduce_mean(
                        -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    )

                    ae_loss = ae_loss + self.config.beta_vae * kl_loss

                total_loss_policy = total_loss_policy + self.config.ae_coeff_policy * ae_loss
                total_loss_policy = total_loss_policy
            
            

        if self.config.use_latent_space_policy:
            
            if self.config.train_ae_policy:

                if VERBOSE: 
                    logger.debug(f"total_loss_policy.dtype={total_loss_policy.dtype}")
                    logger.debug(f"total_loss_policy.shape={total_loss_policy.shape}")
                    logger.debug(f"total_loss_policy={total_loss_policy}")
                    

                    for idx, v in enumerate(self.policy.trainable_variables):
                        logger.debug(f"self.policy.trainable_variables[{idx}].dtype={self.policy.trainable_variables[idx].dtype}")

                    for idx, v in enumerate(self.encoder.trainable_variables):
                        logger.debug(f"self.encoder.trainable_variables[{idx}].dtype={self.encoder.trainable_variables[idx].dtype}")

                    for idx, v in enumerate(self.decoder.trainable_variables):
                        logger.debug(f"self.decoder.trainable_variables[{idx}].dtype={self.decoder.trainable_variables[idx].dtype}")


                policy_grads = tape.gradient(
                    total_loss_policy,
                    self.policy.trainable_variables + self.encoder.trainable_variables + self.decoder.trainable_variables
                )

                if VERBOSE: 
                    logger.debug(f"GRADIENTS ARE CALCULATED:")

                    for idx, g in enumerate(policy_grads):
                        logger.debug(f"policy_grads[{idx}].dtype={policy_grads[idx].dtype}")

                self.policy_optimizer.apply_gradients(zip(
                    policy_grads, 
                    self.policy.trainable_variables + self.encoder.trainable_variables + self.decoder.trainable_variables
                ))

            # Not training decoder, but training encoder
            elif self.config.train_enc_policy:
                policy_grads = tape.gradient(
                    total_loss_policy,
                    self.policy.trainable_variables + self.encoder.trainable_variables
                )

                self.policy_optimizer.apply_gradients(zip(
                    policy_grads, 
                    self.policy.trainable_variables + self.encoder.trainable_variables
                ))

        # Not training AE at all, only policy
        else:
            policy_grads = tape.gradient(
                total_loss_policy,
                self.policy.trainable_variables
            )

            self.policy_optimizer.apply_gradients(zip(
                policy_grads, 
                self.policy.trainable_variables
            ))

        # free memory
        del total_loss_policy, tape

    def replay(self, states, actions, rewards, predictions, dones, next_states):

        # logger.debug(f"REPLAY FN CALLED.")
        # logger.debug(f"len(states) = {len(states)}")
        # logger.debug(f"states[0].shape = {states[0].shape}")
        # logger.debug(f"np.vstack(states).shape = {np.vstack(states).shape}")


        if self.config.use_latent_space_critic:
            if self.config.variational_ae:
                _, _, z = self.encoder( np.vstack(states) )
                _, _, next_z = self.encoder( np.vstack(next_states) )
                values      = self.critic(z)
                next_values = self.critic(next_z)
            else:
                z = self.encoder( np.vstack(states) )
                next_z = self.encoder( np.vstack(next_states) )
                values      = self.critic(z)
                next_values = self.critic(next_z)

        else:
            values      = self.critic(np.vstack(states))
            next_values = self.critic(np.vstack(next_states))
        

        advantages, target = self.get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values),
            discount_factor = self.config.discount_factor, 
            gae_lambda = self.config.gae_lambda,
            normalize = self.config.normalize_gae
        )
       
        kekseethe = list(zip(
            np.vstack(states),
            np.vstack(next_states),
            np.vstack(actions),
            np.vstack(predictions),
            np.vstack(values),
            np.vstack(next_values),
            np.vstack(advantages),
            np.vstack(target)
        ))
       

        for ee in range(self.config.training_epochs):
           
            random.shuffle(kekseethe)
            logger.info(f"[Agent {self.agent_id}] replay epoch {ee}...")

            for step in tqdm(range(len(kekseethe) // self.config.training_minibatch)):
        
                batch  = random.sample(
                    kekseethe, self.config.training_minibatch
                )

                states = np.array([b[0] for b in batch])
                next_states = np.array([b[1] for b in batch])
                actions = np.vstack([b[2] for b in batch])
                predictions = np.vstack([b[3] for b in batch])
                values  = np.vstack([b[4] for b in batch])
                next_values  = np.vstack([b[5] for b in batch])
                advantages  = np.vstack([b[6] for b in batch])
                target = np.vstack([b[7] for b in batch])
                

                if VERBOSE: 

                    for k in range(len(batch[0])):
                        logger.debug(f"batch[0][{k}].shape = {batch[0][k].shape}")

                    logger.debug(f"states.shape = {states.shape}")
                    logger.debug(f"next_states.shape = {next_states.shape}")
                    logger.debug(f"actions.shape = {actions.shape}")
                    logger.debug(f"predictions.shape = {predictions.shape}")
                
                    logger.debug(f"advantages.shape = {advantages.shape}")
                    logger.debug(f"target.shape = {target.shape}")

                    logger.debug(f"values.shape = {values.shape}")
                    logger.debug(f"next_values.shape = {next_values.shape}")

                    logger.debug(f"(actions*predictions).shape = {(actions*predictions).shape}")

                    logger.debug("---------------------------------------------------------------------")

                    N=3
                    
                    logger.debug(f"states[:{N}]:\n{states[:N]}")
                    logger.debug(f"next_states[:{N}]:\n{next_states[:N]}")
                    logger.debug(f"actions[:{N}]:\n{actions[:N]}")
                    logger.debug(f"predictions[:{N}]:\n{predictions[:N]}")
                
                    logger.debug(f"advantages[:{N}]:\n{advantages[:N]}")
                    logger.debug(f"target[:{N}]:\n{target[:N]}")

                    logger.debug(f"values[:{N}]:\n{values[:N]}")
                    logger.debug(f"next_values[:{N}]:\n{next_values[:N]}")

                    logger.debug(f"(actions*predictions)[:{N}] = {(actions*predictions)[:N]}")
                
                self.optimize_critic(states, next_states, actions, predictions, values, next_values, advantages, target)
                
                self.optimize_policy(states, next_states, actions, predictions, values, next_values, advantages, target)
                gc.collect()

                if VERBOSE:
                    logger.info("EXITING AFTER OPT STEP SUCCESSFUL.")
                    sys.exit(0)
            gc.collect()
        gc.collect()
        self.replay_count += 1
        

    def save_checkpoint(self):

        logger.info(f"[Agent {self.agent_id}] episode: {self.episode}/{self.config.episodes}, creating checkpoint...")

        ckpt_path = os.path.join(self.save_path, "checkpoint")

        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
        os.makedirs(ckpt_path)

        with open(os.path.join(ckpt_path, "info.txt"), "w") as f:
            f.write(str({
                "episode": self.episode
            }))
            f.close()

        shutil.copyfile(
            os.path.join(self.save_path, "progress.csv"),
            os.path.join(ckpt_path, "progress_checkpoint.csv")
        )

        # Save models
        if self.encoder: 
            self.encoder.save_weights(os.path.join(ckpt_path, "encoder"))

        if self.decoder: 
            self.decoder.save_weights(os.path.join(ckpt_path, "decoder"))

        self.critic.save_weights(os.path.join(ckpt_path, "critic"))
        self.policy.save_weights(os.path.join(ckpt_path, "policy"))


        with open( os.path.join(ckpt_path, "config.txt"), "w") as f:
            f.write(str(
                self.config.__dict__
            ))
            f.close()
            
        logger.info(f"[Agent {self.agent_id}] episode: {self.episode}/{self.config.episodes}, checkpoint created.")
        
    def load_checkpoint(self, ckpt_path):
        # self.Actor.Actor.load_weights(self.Actor_name)
        # self.Critic.Critic.load_weights(self.Critic_name)
        pass

    def save(self):
        # self.Actor.Actor.save_weights(self.Actor_name)
        # self.Critic.Critic.save_weights(self.Critic_name)
        pass
        
    def run_multiprocesses(self):

        logger.info("Collecting data using run_multiprocesses...")

        works, parent_conns, child_conns = {}, {}, {}
        
        for idx in range(self.config.env_workers):
            parent_conn, child_conn = Pipe()
            
            work = Environment(
                env_idx = idx,
                child_conn = child_conn,
                config = self.config,
                action_size = self.env.action_space.n, 
                visualize=False
            )

            work.start()

            logger.info(f"Started worker {idx}.")

            works[idx] = work
            parent_conns[idx] = parent_conn
            child_conns[idx] = child_conn

        states =        [[] for _ in range(self.config.env_workers)]
        next_states =   [[] for _ in range(self.config.env_workers)]
        actions =       [[] for _ in range(self.config.env_workers)]
        rewards =       [[] for _ in range(self.config.env_workers)]
        dones =         [[] for _ in range(self.config.env_workers)]
        predictions =   [[] for _ in range(self.config.env_workers)]
        score =         [0 for _ in range(self.config.env_workers)]
        eplen =         [0 for _ in range(self.config.env_workers)]

        state = [0 for _ in range(self.config.env_workers)]
        
        logger.debug(f"list(parent_conns.keys()) = {list(parent_conns.keys())}")
        logger.debug(f"list(works.keys()) = {list(works.keys())}")
        logger.debug(f"list(child_conns.keys()) = {list(child_conns.keys())}")

        for worker_id in parent_conns.keys():
            state[worker_id] = parent_conns[worker_id].recv()
            logger.debug(f"state[worker_id].shape = {state[worker_id].shape}")
        
        self.episode = self.episode_start
        while self.episode < self.config.episodes:
            
            # logger.debug(f"state[0].shape={state[0].shape}")
            
            _state = np.reshape(state, [self.config.env_workers, *(state[0].shape[1:])])
            
            # logger.debug(f"_state.shape={_state.shape}")
            
            if self.config.use_latent_space_policy:
                
                if self.config.variational_ae:
                    _, _, z = self.encoder(
                        _state
                    )

                    predictions_list = self.policy(
                        z
                    ).numpy()
                else:
                    z = self.encoder(
                        _state
                    )

                    predictions_list = self.policy(
                        z
                    ).numpy()
            else:
                predictions_list = self.policy(_state).numpy()
            
            actions_list = [np.random.choice(self.env.action_space.n, p=i) for i in predictions_list]

            for worker_id in parent_conns.keys():
                
                # Send action to connection
                parent_conns[worker_id].send(actions_list[worker_id])
                
                action_onehot = np.zeros([self.env.action_space.n])
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

                if done:
                    # average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    logger.info(f"[Agent {self.agent_id}] episode: {self.episode}/{self.config.episodes}, worker: {worker_id}, eplen: {eplen[worker_id]}, score: {score[worker_id]}|{datetime.now()}")

                    with open(os.path.join(self.save_path, f"progress.csv"), 'a') as f:
                        f.write(f"{self.episode}|{worker_id}|{eplen[worker_id]}|{score[worker_id]}|{datetime.now()}\n")
                        f.close()
                    
                    score[worker_id] = 0
                    eplen[worker_id] = 0
                    if(self.episode < self.config.episodes):
                        self.episode += 1
                        
                    if(self.episode > 0 and self.episode % self.config.save_freq == 0):
                        self.save_checkpoint()

            for worker_id in works.keys():
                if len(states[worker_id]) >= self.config.replay_batch:
                    logger.info(f"[Agent {self.agent_id}] [worker {worker_id}] Replaying...")
                    self.replay(
                        states[worker_id], 
                        actions[worker_id],
                        rewards[worker_id],
                        predictions[worker_id],
                        dones[worker_id],
                        next_states[worker_id]
                    )
                    
                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    predictions[worker_id] = []
                    logger.info(f"[Agent {self.agent_id}] [worker {worker_id}] Collecting data...")

        # terminating processes after while loop
        # works.append(work)
        
        for key in works.keys():
            works[key].terminate()
            logger.info(f'PROCESS TERMINATED: {works[key]}')
            works[key].join()

if __name__ == "__main__":

    

    save_path = os.path.join(args.save_path, str(args.agent_id))
    
    if not args.continued:
        if os.path.exists(save_path):
            logger.critical(f'Save path already in use: {save_path}')
            sys.exit(1)
        
        os.makedirs(save_path)

        agent = PPOAgent(config, save_path, args.agent_id, continued=False)
    else:
        agent = PPOAgent(config, save_path, args.agent_id, continued=True)

    if config.profile:

        import warnings

        profiler_options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
        )

        # logdirpath = os.path.join(args.save_path, str(args.agent_id), "tf_logdir")
        # os.makedirs(logdirpath)
        logdirpath = "logdir"
        
        
        warnings.warn(f"TENSORFLOW PROFILE MODE IS TURNED ON!\nLOGS SAVED TO {logdirpath}")


        tf.profiler.experimental.start(
            logdirpath, 
            options=profiler_options
        )

        agent.run_multiprocesses()

        tf.profiler.experimental.stop()
    else:
        agent.run_multiprocesses()
