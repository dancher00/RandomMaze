import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


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
        self.state_space = self._generate_state_space()

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
    
    def transition_function(self, state, action):
        '''
        For given action and state returns list of possible next states and corresponding rewards with their probabilities
        '''
        transitions = []
        # prob, next_state, reward, done
        if action == 4:
            next_state = state
            reward = 0 if next_state == self.goal else -1
            transitions.append([1, next_state, reward, False])
            return transitions
        else:
            intended_move = self.transitions[action]
            next_state = (state[0] + intended_move[0], state[1] + intended_move[1])

            if not self._is_valid(next_state):
                free_neighbors = self._get_free_neighbors(state)
                if free_neighbors:
                    for free_state in free_neighbors:
                        reward = 0 if state == self.goal else -1
                        transitions.append([1 / len(free_neighbors), free_state, reward, False])
                    return transitions
                else:
                    next_state = state
                    reward = 0 if next_state == self.goal else -1
                    transitions.append([1, next_state, reward, False])
                    return transitions

            free_neighbors = self._get_free_neighbors(state)
            for free_state in free_neighbors:
                reward = 0 if free_state == self.goal else -1
                prob = 0.15 / len(free_neighbors) if free_state != next_state else 0.15 / len(free_neighbors) + 0.85
                transitions.append([prob, free_state, reward, False])
            return transitions
    
    def _generate_state_space(self):
        states = []
        for i in range(self.n):
            for j in range(self.m):
                if self.maze[i, j] == 0:
                    states.append((i,j))
        return states

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
