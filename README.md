# Flappy Bird RL with Deep Q-Networks

Autonomous agent to play Flappy Bird without manually programming the game's logic. Unlike supervised learning, where labeled examples are provided, reinforcement learning enables the agent to learn by interacting with the environment, receiving rewards for good actions and penalties for poor ones. Over many episodes, the agent learns a policy that maximizes long-term cumulative reward.

Reinforcement learning implementation of Flappy Bird using Deep Q-Networks (DQN) training pipeline in PyTorch. This includes Experience Replay Buffer, Target Network synchronization, Epsilon-Greedy exploration, Hyperparameter configuration using YAML.
<div align="center">
<img width="216" height="260" alt="flappy-bird" src="https://github.com/user-attachments/assets/933afc63-7546-48e8-8c75-787e4f4bd1e0"/>
</div>

## Overview

1. State Space :
    - The agent receives the following observations:
    - Bird vertical position
    - Bird vertical velocity
    - Horizontal distance to next pipe
    - Vertical distance from pipe gap

2. Action Space :
    - 0 -> DO Nothing
    - 1 -> Flap

3. Reward Function :
    +1   Pass a pipe ; 
    +0.1 Stay alive ;
    -1   Collision ;

---

- Reinforcement Learning with Deep-Q-Net Pipeline :   
```
Enviornment State
   ▼
Deep Neural Network - Predicts Q Values
   ▼
Q-values - Q(Do Nothing)=1.3 / Q(Flap)=2.8 
   ▼
Choose Action - DoNothing / Flap
   ▼
Environment - Returns {state, action, reward, next_state, done}
   ▼
Store Experience in DEQUE as experience replays
   ▼
Replay Memory 
   ▼
Mini-batch Training - Learn in Batches instead of sequentially.
   ▼
Target Network computes target Q-Values using Policy Network.
   ▼
Backpropagate : Calculate Huber Loss and Update Policy Network
```
## Neural Network 
```
Input Layer ➡️ Hidden Layer (128 ReLU) ➡️ Hidden Layer (128 ReLU) ➡️ Output Layer (Q-values - 2)

```
## Features

- Neural network approximates/predicts Q-values for state-action pairs.
- Implemented a replay buffer from scratch using deque to store agent experiences and perform randomized mini-batch sampling for stable training.
- Epsilon-greedy introduces random actions during training to encourage exploration. As epsilon decays, the agent gradually shifts from exploration to exploitation of what it has learned.
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
## Demo Training & Results:
<img width="520" height="262.125" alt="{BC9C9C27-6167-4CBA-B1C2-E9292E785047}" src="https://github.com/user-attachments/assets/3e8f7c51-1df5-48a8-b105-3ec09b46b2b3" />

- Trained for ~16000 Episodes 
- The agent successfully learns obstacle avoidance and demonstrates increasing cumulative rewards over training.
- Although convergence is not yet optimal, the project validates the implementation of Deep Q-Networks and highlights the effects of hyperparameter tuning on training stability.

## Future Work

- Double DQN
- Prioritized Experience Replay
- TensorBoard integration
- Frame stacking
- Reward shaping
- Model evaluation metrics

---
