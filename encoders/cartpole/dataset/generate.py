import gym
import numpy as np
from tqdm import tqdm

env = gym.make('CartPole-v1')

runs = 1000
max_repetitions = 100
obversation_list = []
for _ in tqdm(range(runs)):
    s = env.reset()

    for i in range(max_repetitions):
        s = env.step(np.random.randint(2))
        obversation_list.append(s[0])
        if s[2]:
            break
dataset = np.stack(obversation_list)
np.save('./dataset', dataset)
print(dataset.shape)