import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import tensorflow as tf

from gymnasium.wrappers import GrayScaleObservation


class CropInfoPanel(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = Box(low=0, high=255, shape=(84,84,3), dtype=np.uint8)
    
    def observation(self, observation):
        return observation[0:84, 6:90]

        
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = tf.image.resize(observation, self.shape)
        return observation

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info, None

class IncreaseCarRed(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def observation(self, obs):
        obs = obs.numpy()
        """ Increases the red component to 255 on pixels where r>g && r>b."""
        mask = (obs[:,:,0] > obs[:,:,1]) * (obs[:,:,0] > obs[:,:,2] )
        obs[:,:,0] = mask.astype(np.uint8)*255 + (1-mask.astype(np.uint8))*obs[:,:,0]
        return obs

class RGB2GrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def grayscale(self, colors):
        """Return grayscale of given color."""
        r, g, b = colors
        return 0.21 * r + 0.72 * g + 0.07 * b
        
    def observation(self, obs):
        #r,g,b = obs[:,:,0], obs[:,:,1], obs[:,:,2]
        return np.apply_along_axis(self.grayscale, 2, obs)

class StackFrames(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self._env = env

    def reset(self):
        s0, _ = self._env.reset()

        frames = []
        for i in range(self._skip):
            frames.append(s0)

        return np.stack(frames, axis=2), {}
        
    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        frames = []
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info, _ = self.env.step(action)
            frames.append(obs)
            total_reward += reward
            if done:
                break
        
        return np.stack(frames, axis=2), total_reward, done, info, None