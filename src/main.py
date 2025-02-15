from src.agent import MazeAgent
from src.config import (maze, start, goal, learning_rate,
                        start_epsilon, epsilon_decay,
                        final_epsilon, max_time,
                        discount_factor, n_episodes,
                        plot_path)
from src.environment import MazeEnv
from src.trainer import Trainer
from src.visualizer import MazeVisualizer

if __name__ == "__main__":
    environment = MazeEnv(maze, start, goal, max_time)
    agent = MazeAgent(environment, learning_rate, start_epsilon,
                      epsilon_decay, final_epsilon, discount_factor)

    trainer = Trainer(agent, n_episodes)
    trainer.train()

    visualizer = MazeVisualizer(agent)
    visualizer.display_plots(plot_path)