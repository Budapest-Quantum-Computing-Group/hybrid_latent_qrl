import gymnasium as gym
import numpy as np

import latent_qrl_envs

from tqdm import tqdm
from gymnasium.wrappers import GrayScaleObservation

env = gym.make(
    id='latent_qrl_envs/Maze-v0',
    free_field_color=[59, 179, 0],
    size=48,
    block_size=6,
    target_block_coords=(2, 3),
    target_color=[255, 0, 43],
    randomize_starting_point=False,
    agent_color=[0, 38, 230],
    forbidden_blocks=[(0, 1), (1, 1), (2, 1)],
    forbidden_color=[0, 0, 0]
)

env = GrayScaleObservation(env, keep_dim=True)

runs = 3_000
max_repetitions = 100
observation_list = []
for _ in tqdm(range(runs)):
    s = env.reset()

    for i in range(max_repetitions):
        s = env.step(env.action_space.sample())
        observation_list.append(s[0])
        if s[2]:
            break
print(len(observation_list))
dataset = np.stack(observation_list)
np.save('./dataset', dataset)
print(dataset.shape)

