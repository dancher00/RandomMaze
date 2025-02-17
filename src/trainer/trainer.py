from agent import Agent
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(
            self,
            agent: Agent,
            n_episodes: int
    ):
        self.agent = agent
        self.env = agent.env
        self.n_episodes = n_episodes

    @abstractmethod
    def train(self):
        pass