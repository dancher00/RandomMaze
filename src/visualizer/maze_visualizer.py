import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from agent import QLearningAgent, ValueIterationAgent, PolicyIterationAgent
from visualizer import Visualizer

# Mapping actions to arrows
ACTION_ARROWS = {
    0: "↑",  # Up
    1: "↓",  # Down
    2: "←",  # Left
    3: "→",  # Right
    4: "•"  # Stay
}


class QLearningVisualizer(Visualizer):
    def __init__(
            self,
            agent: QLearningAgent):
        super().__init__(agent)
        self.env = agent.env.unwrapped
        self.df_q_values = None
        self.df_policy = None
        self.extract_q_policy()

    def display_plots(self, plot_path=None):
        super().display_plots(plot_path)
        _, ax = plt.subplots(1, 2, figsize=(16, 8))
        self.plot_results(ax)
        plt.tight_layout()
        if plot_path is None:
            plt.show()
        else:
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(plot_path + "/results.png")

    def extract_q_policy(self):
        """Extracts the best action and Q-values from Q-table for each state."""
        policy_data = []
        q_values_data = []
        for x in range(self.env.n):
            for y in range(self.env.m):
                state = (x, y)
                if self.agent.env.unwrapped.maze[x, y] == 1:
                    best_action = "#"  # Wall
                    best_q_value = None
                else:
                    best_action_idx = np.argmax(self.agent.q_values[state])
                    best_action = ACTION_ARROWS[best_action_idx]
                    best_q_value = np.max(self.agent.q_values[state])
                policy_data.append([x, y, best_action])
                q_values_data.append([x, y, best_q_value])

        self.df_policy = pd.DataFrame(policy_data, columns=["X", "Y", "Action"])
        self.df_q_values = pd.DataFrame(q_values_data, columns=["X", "Y", "Q_Value"])

    def plot_results(self, ax: plt.Axes):
        """Plots the final maze state and overlays Q-values with learned policy on one heatmap."""
        # Create Final State visualization
        display_maze = self.env.maze.copy().astype(float)
        display_maze[self.env.maze == 1] = -100  # Walls
        display_maze[self.env.state] = 100  # Agent position
        display_maze[self.env.goal] = 200  # Goal position

        # Draw agent as a triangle
        agent_x, agent_y = self.env.state
        triangle = patches.RegularPolygon((agent_y, agent_x), numVertices=3, radius=0.3, color='red')

        ax[0].imshow(display_maze)
        ax[0].add_patch(triangle)  #agent
        ax[0].axis("off")
        ax[0].set_title("Last Frame")

        df_q_values_pivot = self.df_q_values.pivot(index="X", columns="Y", values="Q_Value")

        # Draw Q-values as heatmap
        sns.heatmap(df_q_values_pivot, cmap="Greens", annot=False, linewidths=0.5, linecolor='black', ax=ax[1],
                    cbar=True)

        # Overlay walls, goal, and arrows
        for x in range(self.env.n):
            for y in range(self.env.m):
                if self.env.maze[x, y] == 1:
                    rect = patches.Rectangle((y, x), 1, 1, facecolor='black', edgecolor='black')
                    ax[1].add_patch(rect)
                elif (x, y) == self.env.goal:
                    rect = patches.Rectangle((y, x), 1, 1, facecolor='gold', edgecolor='black')
                    ax[1].add_patch(rect)

        # Draw policy arrows
        for _, row in self.df_policy.iterrows():
            x, y, action = row
            if action != "#":  # Ignore walls
                ax[1].text(y + 0.5, x + 0.5, action, ha='center', va='center', fontsize=16, color='red')

        ax[1].set_title("Learned Q-values & Policy")
        return ax


class ValueIterationVisualizer(Visualizer):
    def __init__(self, agent: ValueIterationAgent):
        super().__init__(agent)
        self.agent = agent
        self.env = agent.env
        self.df_state_values = None
        self.df_policy = None
        self.extract_policy()

    def display_plots(self, plot_path=None):
        fig, ax = plt.subplots(figsize=(16, 8))
        # Левый subplot: график метрик (дельты)
        if hasattr(self.agent, 'training_deltas') and self.agent.training_deltas:
            ax.plot(self.agent.training_deltas, marker='o')
            ax.set_title("Convergence (Delta)")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Delta")
        else:
            ax.text(0.5, 0.5, "No training metrics", ha='center', va='center')
        fig.tight_layout()
        if plot_path is None:
            plt.show()
        else:
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(os.path.join(plot_path, "training.png"))

    def extract_policy(self):
        policy_data = []
        values_data = []
        for x in range(self.env.n):
            for y in range(self.env.m):
                state = (x, y)
                if self.env.maze[x, y] == 1:
                    best_action = "#"  # Стена
                    value = None
                else:
                    best_action_idx = self.agent.get_action(state)
                    best_action = ACTION_ARROWS[best_action_idx]
                    value = self.agent.value_function[state]
                policy_data.append([x, y, best_action])
                values_data.append([x, y, value])
        self.df_policy = pd.DataFrame(policy_data, columns=["X", "Y", "Action"])
        self.df_state_values = pd.DataFrame(values_data, columns=["X", "Y", "Value"])

    def plot_results(self, ax: plt.Axes):
        env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        display_maze = env.maze.copy().astype(float)
        display_maze[env.maze == 1] = -100  # walls
        display_maze[env.state] = 100  # agent's position
        display_maze[env.goal] = 200  # goal

        agent_x, agent_y = env.state
        triangle = patches.RegularPolygon((agent_y, agent_x), numVertices=3, radius=0.3, color='red')

        ax.imshow(display_maze)
        ax.add_patch(triangle)
        ax.axis("off")
        ax.set_title("Final State")

        for x in range(env.n):
            for y in range(env.m):
                if env.maze[x, y] == 1:
                    rect = patches.Rectangle((y, x), 1, 1, facecolor='black', edgecolor='black')
                    ax.add_patch(rect)
                elif (x, y) == env.goal:
                    rect = patches.Rectangle((y, x), 1, 1, facecolor='gold', edgecolor='black')
                    ax.add_patch(rect)
        for _, row in self.df_policy.iterrows():
            x, y, action = row
            if action != "#":
                ax.text(y + 0.5, x + 0.5, action, ha='center', va='center', fontsize=16, color='red')
        ax.set_title("Learned State Values & Policy")


class PolicyIterationVisualizer:
    def __init__(self, agent: PolicyIterationAgent):
        self.agent = agent
        self.env = agent.env
        self.df_state_values = None
        self.df_policy = None
        self.extract_policy()

    def display_plots(self, plot_path=None):
        fig, ax = plt.subplots(figsize=(16, 8))
        if hasattr(self.agent, 'policy_changes') and self.agent.policy_changes:
            ax.plot(self.agent.policy_changes, marker='o')
            ax.set_title("Policy Changes per Iteration")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Number of Changes")
        else:
            ax.text(0.5, 0.5, "No training metrics", ha='center', va='center')
        fig.tight_layout()
        if plot_path is None:
            plt.show()
        else:
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(os.path.join(plot_path, "training.png"))

    def extract_policy(self):
        policy_data = []
        values_data = []
        for x in range(self.env.n):
            for y in range(self.env.m):
                state = (x, y)
                if self.env.maze[x, y] == 1:
                    best_action = "#"  # Стена
                    value = None
                else:
                    best_action = ACTION_ARROWS[self.agent.policy.get(state, 4)]
                    value = self.agent.state_values.get(state, 0)
                policy_data.append([x, y, best_action])
                values_data.append([x, y, value])
        self.df_policy = pd.DataFrame(policy_data, columns=["X", "Y", "Action"])
        self.df_state_values = pd.DataFrame(values_data, columns=["X", "Y", "Value"])

    def plot_results(self, ax: plt.Axes):
        env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        display_maze = env.maze.copy().astype(float)
        display_maze[env.maze == 1] = -100
        display_maze[env.state] = 100
        display_maze[env.goal] = 200

        agent_x, agent_y = env.state
        triangle = patches.RegularPolygon((agent_y, agent_x), numVertices=3, radius=0.3, color='red')

        ax.imshow(display_maze)
        ax.add_patch(triangle)
        ax.axis("off")
        ax.set_title("Final State")

        for x in range(env.n):
            for y in range(env.m):
                if env.maze[x, y] == 1:
                    rect = patches.Rectangle((y, x), 1, 1, facecolor='black', edgecolor='black')
                    ax.add_patch(rect)
                elif (x, y) == env.goal:
                    rect = patches.Rectangle((y, x), 1, 1, facecolor='gold', edgecolor='black')
                    ax.add_patch(rect)
        for _, row in self.df_policy.iterrows():
            x, y, action = row
            if action != "#":
                ax.text(y + 0.5, x + 0.5, action, ha='center', va='center', fontsize=16, color='red')
        ax.set_title("Learned State Values & Policy")
