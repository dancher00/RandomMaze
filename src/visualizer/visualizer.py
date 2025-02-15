import matplotlib.pyplot as plt
import numpy as np

from src.agent import MazeAgent


class Visualizer:
    def __init__(
            self,
            agent: MazeAgent):
        self.agent = agent
        self.wrapped_env = agent.env

    def display_plots(self, plot_path=None):
        _, ax = plt.subplots(1, 3, figsize=(20, 8))
        self.plot_training(ax)
        plt.tight_layout()
        if plot_path is None:
            plt.show()
        else:
            plt.savefig(plot_path + "/training.png")

    def plot_training(self, ax: plt.Axes):
        ax[0].plot(np.convolve(self.wrapped_env.return_queue, np.ones(100)))
        ax[0].set_title("Episode Rewards")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Reward")

        ax[1].plot(np.convolve(self.wrapped_env.length_queue, np.ones(100)))
        ax[1].set_title("Episode Lengths")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Length")

        ax[2].plot(np.convolve(self.agent.training_error, np.ones(100)))
        ax[2].set_title("Training Error")
        ax[2].set_xlabel("Episode")
        ax[2].set_ylabel("Temporal Difference")

        return ax