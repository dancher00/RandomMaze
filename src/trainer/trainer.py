from tqdm import tqdm

from agent import MazeAgent, ValueIterationAgent, PolicyIterationAgent


class Trainer:
    def __init__(
            self,
            agent: MazeAgent,
            n_episodes: int
    ):
        self.agent = agent
        self.env = agent.env
        self.n_episodes = n_episodes

    def train(self):
        for episode in tqdm(range(self.n_episodes)):
            obs, info = self.env.reset()
            done = False

            # play one episode
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # update the agent
                self.agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            self.agent.decay_epsilon()


class ValueIterationTrainer:
    def __init__(
            self,
            agent: ValueIterationAgent,
            max_episodes: int
    ):
        self.agent = agent
        self.env = agent.env
        self.max_episodes = max_episodes

    def train(self):
        # Train the agent
        for _ in tqdm(range(self.max_episodes)):
            delta = 0
            # Iterate through all states
            for state in self.env.state_space:
                v = self.agent.value_function[state]
                # Update the value function with action-value functions
                self.agent.value_function[state] = self.agent.compute_action_value(state)
                delta = max(delta, abs(v - self.agent.value_function[state]))
            # Stop criteria
            self.agent.training_deltas.append(delta)
            if delta < self.agent.theta:
                break


class PolicyIterationTrainer:
    def __init__(self, agent: PolicyIterationAgent):
        self.agent = agent
        self.env = agent.env

    def train(self):
        iterations = self.agent.policy_iteration()
        print(f"Policy Iteration converged in {iterations} iterations")