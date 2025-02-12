from dataclasses import dataclass
from typing import Set

@dataclass(frozen=True)
class Position:
    """
    Представляет координаты в лабиринте.
    """
    x: int
    y: int

@dataclass
class MazeMap:
    """
    Содержит информацию о текущем состоянии лабиринта.

    Атрибуты:
      - blocked: множество заблокированных клеток.
      - agent_position: текущая позиция агента.
      - goal_position: позиция цели.
    """
    blocked: Set[Position]
    agent_position: Position
    goal_position: Position

@dataclass
class Observation:
    """
    Результат шага среды.

    Атрибуты:
      - reward: полученное вознаграждение.
      - done: флаг завершения эпизода.
      - score: накопленный счёт.
      - step_count: номер текущего шага.
      - map: текущее состояние лабиринта.
    """
    reward: float
    done: bool
    score: int
    step_count: int
    map: MazeMap
