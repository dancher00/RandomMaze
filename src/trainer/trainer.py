from tqdm import tqdm

from agent import MazeAgent


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
