# В src/main.py
from agent import MazeAgent, ValueIterationAgent, PolicyIterationAgent
from config import (maze, start, goal, learning_rate,
                    start_epsilon, epsilon_decay,
                    final_epsilon, max_time,
                    discount_factor, n_episodes, plot_path)
from environment import MazeEnv
from trainer import Trainer, ValueIterationTrainer, PolicyIterationTrainer
from visualizer import MazeVisualizer, ValueIterationVisualizer, PolicyIterationVisualizer
import argparse
import os

if __name__ == "__main__":
    environment = MazeEnv(maze, start, goal, max_time)

    parser = argparse.ArgumentParser(description='Выбор алгоритма обучения.')
    parser.add_argument('--method', type=str, default="q_learning",
                        choices=['q_learning', 'value_iteration', 'policy_iteration'],
                        help='Название метода.')
    args = parser.parse_args()

    # Определяем папку для сохранения результатов в зависимости от выбранного метода
    if args.method == 'q_learning':
        agent = MazeAgent(environment, learning_rate, start_epsilon,
                          epsilon_decay, final_epsilon, discount_factor)
        results_folder = "/app/results/q-learning"
        trainer = Trainer(agent, n_episodes)
        trainer.train()
        from visualizer.maze_visualizer import MazeVisualizer
        visualizer = MazeVisualizer(agent)
        visualizer.display_plots(results_folder)
        from visualizer.agent_animation import AgentAnimationVisualizer
        anim = AgentAnimationVisualizer(agent)
        anim.create_gif(os.path.join(results_folder, "animation.gif"))
    
    elif args.method == 'value_iteration':
        agent = ValueIterationAgent(environment, discount_factor=discount_factor)
        results_folder = "/app/results/value-iteration"
        trainer = ValueIterationTrainer(agent, 100)
        trainer.train()
        from visualizer.maze_visualizer import ValueIterationVisualizer
        visualizer = ValueIterationVisualizer(agent)
        visualizer.display_plots(results_folder)
        from visualizer.agent_animation import AgentAnimationVisualizer
        anim = AgentAnimationVisualizer(agent)
        anim.create_gif(os.path.join(results_folder, "animation.gif"))

    elif args.method == 'policy_iteration':
        agent = PolicyIterationAgent(environment, discount_factor=discount_factor, theta=1e-2)
        results_folder = "/app/results/policy-iteration"
        trainer = PolicyIterationTrainer(agent)
        trainer.train()
        from visualizer.maze_visualizer import PolicyIterationVisualizer
        visualizer = PolicyIterationVisualizer(agent)
        visualizer.display_plots(results_folder)
        from visualizer.agent_animation import AgentAnimationVisualizer
        anim = AgentAnimationVisualizer(agent)
        anim.create_gif(os.path.join(results_folder, "animation.gif"))
