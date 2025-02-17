# Slippery Random Maze Reinforcement Learning Project

This project is about training an agent to escape "slippery" labyrinths. 
The environment is a grid-based maze where the agent experiences "slippery" dynamics: 
intended moves have an 85% chance of success and a 15% chance of moving to a random adjacent unblocked cell.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation and Usage](#installation-and-usage)
- [License](#license)
- [Demo](#demo)

## Overview

### Environment

The Slippery Random Maze project simulates a grid-based maze environment where an agent attempts to navigate from a starting position to a goal while encountering stochastic dynamics. The agent can move up, down, left, right, or stay in place. The movement action succeeds with an 85% probability if the target cell is not blocked; otherwise, the remaining 15% is equally distributed among other available adjacent cells.

Python code for state space:

```
states = []
for i in range(self.n):
    for j in range(self.m):
        if self.maze[i, j] == 0:
            states.append((i,j))
```

The agent receives a -1 reward for any action if it does not reach the goal cell as a result. The agent can't hit walls by design. When the goal is reached, the agent receives a reward of 0:

```python
reward = 0 if self.state == self.goal else -1
```

### Agents

Three reinforcement learning algorithms are implemented in this environment: value iteration, policy iteration and q-learning. For first two approaches state transition function is used to calculate the probability of next state, for q-learning method next state is sampled from distribution described above.

Q-function update:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

State value function update:

$$
V(s) = \sum_{s'} P(s' \mid s, \pi(s)) \left[ r(s, \pi(s), s') + \gamma V(s') \right]
$$

## Installation and Usage

**Clone the repository:**

```
git clone https://github.com/dancher00/Slippery-Random-Maze.git
cd Slippery-Random-Maze
```

Use docker
```
docker build --no-cache -t randommaze .

./run.sh q_learning
./run.sh value_iteration
./run.sh policy_iteration
```

Use `src/config.py` to modify parameters of the training process.
## Results

Policy Iteration Training
![PI_train](https://github.com/dancher00/Slippery-Random-Maze/blob/main/results/policy-iteration/training.png)

Value Iteration Training
![VI_train](https://github.com/dancher00/Slippery-Random-Maze/blob/main/results/value-iteration/training.png)

Q-Learning Training
![Q_train](https://github.com/dancher00/Slippery-Random-Maze/blob/main/results/q-learning/training.png)

Policy Iteration Results
![PI_results](https://github.com/dancher00/Slippery-Random-Maze/blob/main/results/policy-iteration/results.png)

Value Iteration Results
![VI_results](https://github.com/dancher00/Slippery-Random-Maze/blob/main/results/value-iteration/results.png)

Q-Learning Results
![Q_results](https://github.com/dancher00/Slippery-Random-Maze/blob/main/results/q-learning/results.png)

## Demo
Below is a demonstration of the project in action:

![Slippery Maze Demo](https://github.com/dancher00/Slippery-Random-Maze/blob/main/demo_maze.gif)

## License
This project is licensed under the MIT License.
