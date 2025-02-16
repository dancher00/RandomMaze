from agent import MazeAgent, ValueIterationAgent
from config import (maze, start, goal, learning_rate,
                        start_epsilon, epsilon_decay,
                        final_epsilon, max_time,
                        discount_factor, n_episodes,
                        plot_path)
from environment import MazeEnv
from trainer import Trainer, ValueIterationTrainer
from visualizer import MazeVisualizer, ValueIterationVisualizer
import argparse

if __name__ == "__main__":
    environment = MazeEnv(maze, start, goal, max_time)

    parser = argparse.ArgumentParser(description='Process some strings.')
    parser.add_argument('--method', type=str, default="q_learning", choices=['q_learning', 'value_iteration', 'policy_iteration'], help='The name of the method.')

    # Parse the arguments
    args = parser.parse_args()
    if args.method == 'q_learning':
        agent = MazeAgent(environment, learning_rate, start_epsilon,
                        epsilon_decay, final_epsilon, discount_factor)

        plot_path = "/app/img"

        trainer = Trainer(agent, n_episodes)
        trainer.train()

        visualizer = MazeVisualizer(agent)
        visualizer.display_plots(plot_path)
    
    elif args.method == 'value_iteration':
        agent = ValueIterationAgent(environment)

        plot_path = "/app/img"

        trainer = ValueIterationTrainer(agent, 100)
        trainer.train()

        visualizer = ValueIterationVisualizer(agent)
        visualizer.display_plots(plot_path)

    elif args.method == 'policy_iteration':
        pass
