from agent import MazeAgent
from config import (maze, start, goal, learning_rate,
                        start_epsilon, epsilon_decay,
                        final_epsilon, max_time,
                        discount_factor, n_episodes,
                        plot_path)
from environment import MazeEnv
from trainer import Trainer
from visualizer import MazeVisualizer

if __name__ == "__main__":
    environment = MazeEnv(maze, start, goal, max_time)
    agent = MazeAgent(environment, learning_rate, start_epsilon,
                      epsilon_decay, final_epsilon, discount_factor)

    plot_path = "/app/img"

    trainer = Trainer(agent, n_episodes)
    trainer.train()

    visualizer = MazeVisualizer(agent)
    visualizer.display_plots(plot_path)