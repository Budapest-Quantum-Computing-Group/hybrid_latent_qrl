import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RGBMazeEnv(gym.Env):
    def __init__(self, free_field_color, size, block_size, target_block_coords, target_color, randomize_starting_point,
                 agent_color, forbidden_blocks, forbidden_color):
        super(RGBMazeEnv, self).__init__()

        assert size % block_size == 0, "Size must be an integer multiple of block_size."

        self.free_field_color = np.array(free_field_color, dtype=np.uint8)
        self.size = size
        self.block_size = block_size
        self.target_block_coords = target_block_coords
        self.target_color = np.array(target_color, dtype=np.uint8)
        self.randomize_starting_point = randomize_starting_point
        self.agent_color = np.array(agent_color, dtype=np.uint8)
        self.forbidden_blocks = forbidden_blocks
        self.forbidden_color = np.array(forbidden_color, dtype=np.uint8)

        self.n_blocks_per_side = self.size // self.block_size

        # Define action and observation space
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)

        # Observation will be the RGB image of the maze
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8)

        self.agent_position = [0, 0]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.maze = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self.maze[:] = self.free_field_color

        # Place target
        self.place_block(self.target_block_coords, self.target_color)

        # Place forbidden blocks
        for block in self.forbidden_blocks:
            self.place_block(block, self.forbidden_color)

        # Initialize or randomize agent position
        if self.randomize_starting_point:
            while True:
                self.agent_position = [np.random.randint(self.n_blocks_per_side),
                                       np.random.randint(self.n_blocks_per_side)]
                if self.agent_position not in self.forbidden_blocks and self.agent_position != self.target_block_coords:
                    break
        else:
            self.agent_position = [0, 0]

        self.update_maze_state()
        return self.maze, {}

    def step(self, action):
        # Proposed new position based on action
        new_position = self.agent_position.copy()
        if action == 0:  # Up
            new_position[1] -= 1
        elif action == 1:  # Down
            new_position[1] += 1
        elif action == 2:  # Left
            new_position[0] -= 1
        elif action == 3:  # Right
            new_position[0] += 1

        # Keep agent within bounds
        self.agent_position = [
            max(0, min(self.n_blocks_per_side - 1, self.agent_position[0])),
            max(0, min(self.n_blocks_per_side - 1, self.agent_position[1]))
        ]

        # Check if new position is off the map or in forbidden blocks
        if (new_position[0] < 0 or new_position[0] >= self.n_blocks_per_side or
                new_position[1] < 0 or new_position[1] >= self.n_blocks_per_side or
                tuple(new_position) in self.forbidden_blocks):
            reward = -1  # Penalize for invalid move
        else:
            self.agent_position = new_position  # Update to valid position
            reward = 1 if self.agent_position == list(self.target_block_coords) else -0.01

        # Update the maze with the new agent position
        self.update_maze_state()

        # Compute reward and done
        done = self.agent_position == list(self.target_block_coords)

        return self.maze, reward, done, False, {}

    def place_block(self, block_coords, color):
        y, x = block_coords
        self.maze[x * self.block_size:(x + 1) * self.block_size, y * self.block_size:(y + 1) * self.block_size] = color

    def update_maze_state(self):
        self.maze[:] = self.free_field_color  # Reset maze
        self.place_block(self.target_block_coords, self.target_color)  # Re-place target
        for block in self.forbidden_blocks:  # Re-place forbidden blocks
            self.place_block(block, self.forbidden_color)
        self.place_block(self.agent_position, self.agent_color)  # Place agent

    def render(self, mode='human'):
        if mode == 'human':
            # For simplicity, this example will not implement a human-viewable rendering.
            # This could be extended
            pass