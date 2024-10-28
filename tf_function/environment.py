from multiprocessing import Process

import gymnasium as gym
import numpy as np
import sys
import tensorflow as tf

import logging

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

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
        with tf.device("/CPU"):
            logger.info(f"[Environment {self.env_idx}]: running...")
            super(Environment, self).run()

            logger.info(f"[Environment {self.env_idx}]: resetting...")
            state, _ = self.env.reset()
            state = self.state_preprocessing(state)
            logger.info(f"[Environment {self.env_idx}]: is reset.")
            logger.info(f"[Environment {self.env_idx}]: max_episode_len = {self.config.max_episode_len}")

            state = np.reshape(state, [1, *state.shape])

            state = state.astype(np.float64)
            self.child_conn.send(state)

            while True:
                action = self.child_conn.recv()

                if self.is_render and self.env_idx == 0:
                    self.env.render()

                state, reward, done, truncation, info = self.env.step(action)
                state = self.state_preprocessing(state)
                state = np.reshape(state, [1, *state.shape])
                state = state.astype(np.float64)

                if self.current_step >= self.config.max_episode_len:
                    done = True

                done = done or truncation

                if done:
                    self.current_step = 0
                    state, _ = self.env.reset()
                    state = self.state_preprocessing(state)
                    state = np.reshape(state, [1, *state.shape])

                self.current_step = self.current_step + 1
                self.child_conn.send([state, reward, done, info])
