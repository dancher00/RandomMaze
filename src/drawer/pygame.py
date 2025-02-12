import pygame
from src.state import MazeMap, Position
from src.drawer.drawer import Drawer

class PygameDrawer(Drawer):
    """
    Отрисовщик среды с использованием Pygame.
    Рисует сетку, заблокированные клетки, позицию агента и цель.
    """
    def __init__(self, grid_size=10, cell_size=40, framerate=10):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.framerate = framerate

        self.clock = pygame.time.Clock()
        pygame.init()
        self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
        pygame.display.set_caption("Slippery Random Maze RL")

    def draw(self, maze_map: MazeMap) -> None:
        # Цвета
        bg_color = (255, 255, 255)       # Белый фон
        blocked_color = (0, 0, 0)        # Чёрные заблокированные клетки
        grid_color = (200, 200, 200)     # Светло-серые линии сетки
        agent_color = (0, 0, 255)        # Синий агент
        goal_color = (0, 255, 0)         # Зелёная цель

        self.screen.fill(bg_color)

        # Рисуем сетку
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, grid_color, rect, 1)

        # Рисуем заблокированные клетки
        for block in maze_map.blocked:
            rect = pygame.Rect(block.y * self.cell_size, block.x * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, blocked_color, rect)

        # Рисуем цель
        goal = maze_map.goal_position
        rect = pygame.Rect(goal.y * self.cell_size, goal.x * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, goal_color, rect)

        # Рисуем агента
        agent = maze_map.agent_position
        cx = agent.y * self.cell_size + self.cell_size // 2
        cy = agent.x * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        pygame.draw.circle(self.screen, agent_color, (cx, cy), radius)

        pygame.display.flip()
        self.clock.tick(self.framerate)

    def close(self) -> None:
        pygame.quit()
