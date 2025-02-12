import random
from src.controller.controller import Controller, ActionSpaceEnum
from src.state import Observation

class BasicController(Controller):
    """
    Базовый контроллер, выбирающий действия случайным образом.
    """
    def __init__(self):
        self.action_space = [
            ActionSpaceEnum.UP,
            ActionSpaceEnum.DOWN,
            ActionSpaceEnum.LEFT,
            ActionSpaceEnum.RIGHT,
            ActionSpaceEnum.STAY
        ]

    def get_action(self, observation: Observation) -> ActionSpaceEnum:
        return random.choice(self.action_space)
