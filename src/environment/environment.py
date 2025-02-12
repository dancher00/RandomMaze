from abc import ABC, abstractmethod
from src.state import Observation

class MazeEnvironment(ABC):
    @abstractmethod
    def reset(self) -> Observation:
        """
        Сброс состояния среды.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Observation:
        """
        Выполнение шага среды с заданным действием.
        """
        pass
