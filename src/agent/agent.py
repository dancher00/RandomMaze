import gymnasium as gym
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(
        self,
        env: gym.Env,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.discount_factor = discount_factor

    @abstractmethod
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        pass
