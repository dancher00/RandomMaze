# В src/visualizer/agent_animation.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import numpy as np
import os

class AgentAnimationVisualizer:
    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env

    def simulate_episode(self):
        states = []
        state, _ = self.env.reset()
        states.append(state)
        done = False
        while not done:
            action = self.agent.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            states.append(next_state)
            done = terminated or truncated
            state = next_state
        return states

    def draw_maze(self, ax, agent_state):
        env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        maze = env.maze.copy().astype(float)
        for x in range(env.n):
            for y in range(env.m):
                if env.maze[x, y] == 1:
                    ax.add_patch(patches.Rectangle((y, x), 1, 1, facecolor='black'))
                else:
                    ax.add_patch(patches.Rectangle((y, x), 1, 1, facecolor='white', edgecolor='gray'))
        goal_x, goal_y = env.goal
        ax.add_patch(patches.Rectangle((goal_y, goal_x), 1, 1, facecolor='gold'))
        agent_x, agent_y = agent_state
        ax.add_patch(patches.Circle((agent_y + 0.5, agent_x + 0.5), 0.3, color='red'))
        ax.set_xlim(0, env.m)
        ax.set_ylim(env.n, 0)
        ax.axis('off')


    def create_gif(self, gif_path="agent_animation.gif", interval=0.5):
        states = self.simulate_episode()
        images = []
        for state in states:
            fig, ax = plt.subplots(figsize=(5, 5))
            self.draw_maze(ax, state)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
            image = image[:, :, :3]
            images.append(image)
            plt.close(fig)
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        imageio.mimsave(gif_path, images, duration=interval)
