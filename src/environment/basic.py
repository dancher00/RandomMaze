import random
from src.environment.environment import MazeEnvironment
from src.state import Position, MazeMap, Observation

class BasicMazeEnvironment(MazeEnvironment):
    """
    Простая реализация лабиринта со скользящей динамикой.
    Агент начинает в верхнем левом углу, цель — в правом нижнем.
    Заблокированные клетки генерируются случайно (кроме старта и цели).
    """
    def __init__(self, grid_size=10, block_prob=0.2, max_steps=100, seed=None):
        self.grid_size = grid_size
        self.block_prob = block_prob
        self.max_steps = max_steps
        if seed is not None:
            random.seed(seed)
        self.reset()

    def reset(self) -> Observation:
        # Генерируем заблокированные клетки случайно
        blocked = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Гарантируем, что старт и цель не заблокированы
                if (i, j) in [(0, 0), (self.grid_size - 1, self.grid_size - 1)]:
                    continue
                if random.random() < self.block_prob:
                    blocked.add(Position(i, j))
        self.blocked = blocked
        self.start = Position(0, 0)
        self.goal = Position(self.grid_size - 1, self.grid_size - 1)
        self.agent_position = self.start
        self.step_count = 0
        self.score = 0
        self.done = False
        self.maze_map = MazeMap(blocked=self.blocked, agent_position=self.agent_position, goal_position=self.goal)
        return Observation(reward=0, done=self.done, score=self.score, step_count=self.step_count, map=self.maze_map)

    def _is_valid(self, pos: Position) -> bool:
        # Проверка выхода за границы и заблокированности
        if pos.x < 0 or pos.x >= self.grid_size or pos.y < 0 or pos.y >= self.grid_size:
            return False
        if pos in self.blocked:
            return False
        return True

    def _get_neighbors(self, pos: Position):
        neighbors = []
        # Рассматриваем 4 направления: вверх, вниз, влево, вправо
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            new_pos = Position(pos.x + dx, pos.y + dy)
            if self._is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def step(self, action: int) -> Observation:
        if self.done:
            return Observation(reward=0, done=True, score=self.score, step_count=self.step_count, map=self.maze_map)
        
        current = self.agent_position
        new_position = current  # по умолчанию остаёмся на месте

        # Если выбрано действие STAY (код 4), агент остаётся на месте
        if action == 4:
            new_position = current
        else:
            # Определяем смещение для выбранного действия
            if action == 0:      # UP
                delta = (-1, 0)
            elif action == 1:    # DOWN
                delta = (1, 0)
            elif action == 2:    # LEFT
                delta = (0, -1)
            elif action == 3:    # RIGHT
                delta = (0, 1)
            else:
                delta = (0, 0)

            intended = Position(current.x + delta[0], current.y + delta[1])
            neighbors = self._get_neighbors(current)
            
            if self._is_valid(intended):
                # Если целевое направление доступно,
                # с вероятностью 85% переходим туда, иначе выбираем одну из других соседних клеток
                other_neighbors = [pos for pos in neighbors if pos != intended]
                if random.random() < 0.85:
                    new_position = intended
                else:
                    if other_neighbors:
                        new_position = random.choice(other_neighbors)
                    else:
                        new_position = intended
            else:
                # Если целевое направление заблокировано или вне лабиринта,
                # распределяем вероятность равномерно между доступными соседями
                if neighbors:
                    new_position = random.choice(neighbors)
                else:
                    new_position = current

        self.agent_position = new_position
        self.maze_map.agent_position = new_position
        self.step_count += 1

        # Штраф за шаг
        reward = -0.1
        # Если достигнута цель, даём бонус и завершаем эпизод
        if new_position == self.goal:
            reward += 100
            self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        self.score += reward

        return Observation(reward=reward, done=self.done, score=self.score, step_count=self.step_count, map=self.maze_map)
