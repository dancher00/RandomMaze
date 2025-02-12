import pygame
from src.environment import BasicMazeEnvironment
from src.drawer import PygameDrawer
from src.controller import BasicController

if __name__ == "__main__":
    grid_size = 10
    cell_size = 40
    max_steps = 100
    framerate = 10

    environment = BasicMazeEnvironment(grid_size=grid_size, block_prob=0.2, max_steps=max_steps, seed=42)
    observation = environment.reset()
    

    drawer = PygameDrawer(grid_size=grid_size, cell_size=cell_size, framerate=framerate)
    controller = BasicController()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = controller.get_action(observation)
        observation = environment.step(action)
        print(f"Step: {observation.step_count}, Action: {action}, Reward: {observation.reward}, Score: {observation.score}")
        drawer.draw(observation.map)

        if observation.done:
            print("Episode finished. Resetting environment.")
            observation = environment.reset()

    drawer.close()
