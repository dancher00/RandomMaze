from collections import defaultdict

import gymnasium as gym
import numpy as np

from agent import Agent
from config import n_episodes
from environment import MazeEnv


class QLearningAgent(Agent):
    def __init__(self, env: MazeEnv, learning_rate: float,
                 initial_epsilon: float, epsilon_decay: float,
                 final_epsilon: float, discount_factor: float = 0.95):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        super().__init__(env, discount_factor)
        self.env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
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


class ValueIterationAgent(Agent):
    def __init__(self, env: MazeEnv, discount_factor=0.95, theta=1e-2):
        super().__init__(env, discount_factor)
        self.env = env
        self.discount_factor = discount_factor  # Discount factor
        self.theta = theta  # Threshold to stop the iteration
        
        # Initialize value function for each state.
        self.value_function = np.zeros(env.maze.shape)
        self.training_deltas = []

    def compute_action_value_(self, state):
        """
        Computes value function for given state
        """
        action_values = []
        for action in range(self.env.action_space.n):
            action_value = 0
            # Process all possible next states and their probabilities and rewards
            for transition in self.env.transition_function(state, action):
                prob, next_state, reward, done = transition
                action_value += prob * (reward + self.discount_factor * self.value_function[next_state])
            action_values.append(action_value)
        return action_values

    def compute_action_value(self, state):
        action_values = self.compute_action_value_(state)
        return max(action_values)
    
    def get_action(self, state):
        action_values = self.compute_action_value_(state)
        return np.argmax(action_values)
    


class PolicyIterationAgent(Agent):
    def __init__(self, env: MazeEnv, discount_factor: float = 0.95, theta: float = 1e-2):
        super().__init__(env, discount_factor)
        self.env = env
        self.theta = theta
        self.policy = {}
        self.state_values = {}
        for state in env.state_space:
            self.policy[state] = np.random.choice(range(self.env.action_space.n))
            self.state_values[state] = 0.0
        self.policy_changes = []

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in self.env.state_space:
                v = self.state_values[state]
                new_v = 0
                action = self.policy[state]
                for transition in self.env.transition_function(state, action):
                    prob, next_state, reward, done = transition
                    new_v += prob * (reward + self.discount_factor * self.state_values.get(next_state, 0))
                self.state_values[state] = new_v
                delta = max(delta, abs(v - new_v))
            if delta < self.theta:
                break


    def policy_improvement(self):
        policy_stable = True
        changes = 0
        for state in self.env.state_space:
            old_action = self.policy[state]
            action_values = []
            for action in range(self.env.action_space.n):
                q = 0
                for transition in self.env.transition_function(state, action):
                    prob, next_state, reward, done = transition
                    q += prob * (reward + self.discount_factor * self.state_values.get(next_state, 0))
                action_values.append(q)
            best_action = int(np.argmax(action_values))
            self.policy[state] = best_action
            if best_action != old_action:
                policy_stable = False
                changes += 1
        return changes

    def policy_iteration(self):
        iterations = 0
        while True:
            self.policy_evaluation()
            changes = self.policy_improvement()
            self.policy_changes.append(changes)
            iterations += 1
            if changes == 0:
                break
        return iterations

    def get_action(self, state: tuple[int, int, bool] | tuple[int, int]):
        return self.policy.get(state, self.env.action_space.sample())

    def simulate_episode(self):
        states = []
        state, _ = self.env.reset()
        states.append(state)
        done = False
        while not done:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            states.append(next_state)
            done = terminated or truncated
            state = next_state
        return states