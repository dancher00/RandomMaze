# В src/main.py
from agent import QLearningAgent, ValueIterationAgent, PolicyIterationAgent
from config import (maze, start, goal, learning_rate,
                    start_epsilon, epsilon_decay,
                    final_epsilon, max_time,
                    discount_factor, n_episodes, plot_path)
from environment import MazeEnv
from trainer import QLearningTrainer, ValueIterationTrainer, PolicyIterationTrainer
from visualizer import QLearningVisualizer, ValueIterationVisualizer, PolicyIterationVisualizer
from visualizer.agent_animation import AgentAnimationVisualizer
import argparse
import os

if __name__ == "__main__":
    environment = MazeEnv(maze, start, goal, max_time)

    parser = argparse.ArgumentParser(description='Algorithm Choosing...')
    parser.add_argument('--method', type=str, default="q_learning",
                        choices=['q_learning', 'value_iteration', 'policy_iteration'],
                        help="method's name...")
    args = parser.parse_args()

    if args.method == 'value_iteration':
        agent = ValueIterationAgent(environment, discount_factor=discount_factor)
        results_folder = "../results/value-iteration"
        trainer = ValueIterationTrainer(agent, n_episodes)
        trainer.train()
        visualizer = ValueIterationVisualizer(agent)
    elif args.method == 'policy_iteration':
        agent = PolicyIterationAgent(environment, discount_factor=discount_factor, theta=1e-2)
        results_folder = "../results/policy-iteration"
        trainer = PolicyIterationTrainer(agent, n_episodes)
        trainer.train()
        visualizer = PolicyIterationVisualizer(agent)
    else:
        agent = QLearningAgent(environment, learning_rate, start_epsilon,
                               epsilon_decay, final_epsilon, discount_factor)
        results_folder = "../results/q-learning"
        trainer = QLearningTrainer(agent, n_episodes)
        trainer.train()
        visualizer = QLearningVisualizer(agent)

    visualizer.display_plots(results_folder)
    anim = AgentAnimationVisualizer(agent)
    anim.create_gif(os.path.join(results_folder, "animation.gif"))