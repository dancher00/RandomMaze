from abc import ABC, abstractmethod
from enum import Enum
from src.state import Observation

class ActionSpaceEnum(int, Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

class Controller(ABC):
    @abstractmethod
    def get_action(self, observation: Observation) -> ActionSpaceEnum:
 
        pass
