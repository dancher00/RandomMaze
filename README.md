# Slippery Random Maze Reinforcement Learning Project

This project is a minimal environment designed for experimenting with reinforcement learning algorithms. The environment is a grid-based maze where the agent experiences "slippery" dynamics: intended moves have an 85% chance of success and a 15% chance of moving to a random adjacent unblocked cell.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Demo](#demo)
- [License](#license)

## Overview

The Slippery Random Maze project simulates a grid-based maze environment where an agent attempts to navigate from a starting position to a goal while encountering stochastic dynamics. The agent can move up, down, left, right, or stay in place. The movement action succeeds with an 85% probability if the target cell is not blocked; otherwise, the remaining 15% is equally distributed among other available adjacent cells.

## Features

- **Grid-based Maze:** Dynamic maze with random blocked cells.
- **Stochastic Dynamics:** Moves have a chance to deviate from the intended direction.
- **Visual Interface:** Rendered using Pygame.
- **Modular Design:** Components for environment, controller, drawer, and state.
- **Extendable:** Easy to integrate with reinforcement learning algorithms.

## Installation

**Clone the repository:**

```
git clone https://github.com/dancher00/Slippery-Random-Maze.git
cd Slippery-Random-Maze
```


## Usage

Use docker

```
docker build --no-cache -t randommaze .


docker run -it -v "$(pwd)/results":/app/results randommaze python main.py --method q_learning
docker run -it -v "$(pwd)/results":/app/results randommaze python main.py --method value_iteration
docker run -it -v "$(pwd)/results":/app/results randommaze python main.py --method policy_iteration

```

## Demo

Below is a demonstration of the project in action:

![Slippery Maze Demo](https://github.com/dancher00/Slippery-Random-Maze/blob/main/demo.gif)


## License
This project is licensed under the MIT License.
