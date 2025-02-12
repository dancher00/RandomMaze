from abc import ABC, abstractmethod
from src.state import MazeMap

class Drawer(ABC):
    @abstractmethod
    def draw(self, map: MazeMap) -> None:
        """
        Отрисовывает текущее состояние лабиринта.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Освобождает ресурсы (например, закрывает окно).
        """
        pass
