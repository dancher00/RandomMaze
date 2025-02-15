import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self, maze, start, goal, max_time=200):
        super(MazeEnv, self).__init__()
        self.maze = np.array(maze)
        self.start = start
        self.goal = goal
        self.state = start
        self.n, self.m = self.maze.shape

        self.action_space = spaces.Discrete(5)  # 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
        self.observation_space = spaces.Tuple((spaces.Discrete(self.n), spaces.Discrete(self.m)))

        self.discount = 0.99
        self.transitions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.time = 0
        self.max_time = max_time

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        self.time = 0
        return self.state, {}

    def step(self, action):
        if action == 4:
            next_state = self.state
        else:
            intended_move = self.transitions[action]
            next_state = (self.state[0] + intended_move[0], self.state[1] + intended_move[1])

            if not self._is_valid(next_state):
                free_neighbors = self._get_free_neighbors(self.state)
                if free_neighbors:
                    next_state = free_neighbors[np.random.choice(len(free_neighbors))]
                else:
                    next_state = self.state
            elif np.random.rand() > 0.85:
                free_neighbors = self._get_free_neighbors(self.state)
                if free_neighbors:
                    next_state = free_neighbors[np.random.choice(len(free_neighbors))]

        self.state = next_state
        reward = 0 if self.state == self.goal else -1
        self.time += 1
        done = self.time >= self.max_time
        return self.state, reward, done, False, {}

    def _is_valid(self, state):
        x, y = state
        return 0 <= x < self.n and 0 <= y < self.m and self.maze[x, y] == 0

    def _get_free_neighbors(self, state):
        neighbors = []
        for dx, dy in self.transitions[:4]:
            new_state = (state[0] + dx, state[1] + dy)
            if self._is_valid(new_state):
                neighbors.append(new_state)
        return neighbors

    def render(self):
        display_maze = self.maze.copy()
        display_maze[self.maze == 1] = -100
        display_maze[self.state] = 100
        display_maze[self.goal] = 200
        plt.imshow(display_maze)
        plt.title('Current Maze')
        plt.show()
