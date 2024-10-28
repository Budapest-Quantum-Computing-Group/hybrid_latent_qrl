from gymnasium.envs.registration import register
from latent_qrl_envs.Maze import RGBMazeEnv


register(
    id="latent_qrl_envs/Maze-v0",
    entry_point="latent_qrl_envs:RGBMazeEnv",
    max_episode_steps=100
)
