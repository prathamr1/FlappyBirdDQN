# Flappy Bird RL with Deep Q-Networks

This repository contains a reinforcement learning implementation of Flappy Bird using Deep Q-Networks (DQN).

## Overview

- Environment: Flappy Bird styled game.
- Algorithm: Deep Q-Network (DQN).
- Framework: PyTorch.
- Python: 3.10.
- Gym: Gymnasium-compatible environment.

## Features

- Neural network approximates Q-values for state-action pairs.
- Experience replay buffer by DEQUE.
- Epsilon-greedy exploration and exploitation.
- Training loop for agent learning.

## Requirements

- Python 3.10
- PyTorch
- Gymnasium
- NumPy

## Installation

1. Create a Python 3.10 virtual environment.
2. Install dependencies:

```bash
conda env create -f environment.yml
```

```bash
pip install torch gymnasium numpy
```

## Usage

Run training with the main script or training entrypoint.
1. Train the agent :
```bash
python agent.py flappybirdsv0 --train
```
2. See the agent in action :
```bash
python agent.py flappybirdsv0 
```

## Notes

- The project is designed to demonstrate how to train an RL agent for Flappy Bird using DQN.
- Adjust hyperparameters for better performance.
- Use Gymnasium-compatible wrappers if needed for environment integration.
