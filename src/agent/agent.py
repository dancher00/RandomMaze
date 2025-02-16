from collections import defaultdict

import gymnasium as gym
import numpy as np

from config import n_episodes
from environment import MazeEnv


class MazeAgent:
    def __init__(
        self,
        env: MazeEnv,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class ValueIterationAgent:
    def __init__(self, env, discount_factor=0.95, theta=1e-2):
        self.env = env
        self.discount_factor = discount_factor  # Discount factor
        self.theta = theta  # Threshold to stop the iteration
        
        # Initialize value function for each state.
        self.value_function = np.zeros(env.maze.shape)

    def compute_action_value(self, state):
        '''
        Computes value function for given state 
        '''
        action_values = []
        for action in range(self.env.action_space.n):
            action_value = 0
            # Process all possible next states and their probabilities and rewards
            for transition in self.env.transition_function(state, action):
                prob, next_state, reward, done = transition
                action_value += prob * (reward + self.discount_factor * self.value_function[next_state])
            action_values.append(action_value)
        return max(action_values)
    
    def get_action(self, state):
        action_values = []
        for action in range(self.env.action_space.n):
            action_value = 0
            for transition in self.env.transition_function(state, action):
                prob, next_state, reward, _ = transition
                action_value += prob * (reward + self.discount_factor * self.value_function[next_state])
            action_values.append(action_value)
    
        return np.argmax(action_values)