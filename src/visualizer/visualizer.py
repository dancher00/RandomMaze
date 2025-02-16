import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from agent import MazeAgent


class Visualizer:
    def __init__(
            self,
            agent: MazeAgent):
        self.agent = agent
        self.wrapped_env = agent.env

    def display_plots(self, plot_path=None):
        _, ax = plt.subplots(1, 3, figsize=(20, 6))
        self.plot_training(ax)
        plt.tight_layout()

        if plot_path is None:
            plt.show()
        else:
            os.makedirs(plot_path, exist_ok=True)  
            plt.savefig(os.path.join(plot_path, "training.png"))

    def plot_training(self, ax: plt.Axes):
        
        sns.set_style("whitegrid")

        sns.lineplot(np.convolve(self.wrapped_env.return_queue, np.ones(100)), color="blue", ax=ax[0])
        ax[0].set_title("Episode Rewards", fontsize=14, fontweight="bold")
        ax[0].set_xlabel("Episode", fontsize=11, fontweight="bold")
        ax[0].set_ylabel("Reward", fontsize=11, fontweight="bold")
        ax[0].grid(True, linestyle="--", alpha=0.5)

        sns.lineplot(np.convolve(self.wrapped_env.length_queue, np.ones(100)), color="green", ax=ax[1])
        ax[1].set_title("Episode Lengths", fontsize=14, fontweight="bold")
        ax[1].set_xlabel("Episode", fontsize=11, fontweight="bold")
        ax[1].set_ylabel("Length", fontsize=11, fontweight="bold")
        ax[1].grid(True, linestyle="--", alpha=0.5)

        sns.lineplot(np.convolve(self.agent.training_error, np.ones(100)), color="red", ax=ax[2])
        ax[2].set_title("Training Error", fontsize=14, fontweight="bold")
        ax[2].set_xlabel("Episode", fontsize=11, fontweight="bold")
        ax[2].set_ylabel("Temporal Difference", fontsize=11, fontweight="bold")
        ax[2].grid(True, linestyle="--", alpha=0.5)

        return ax