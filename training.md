#### Steps to train DQN flappy bird:

1. Setup enviornment
2. Define DQN 
3. Create experiance replay 
4. multiple episodes for DQN train Setup
5. Epsilon greedy policy + Hyperparameter tune
6. Target network 
8. Train and Test DQN

---

Device --> NVIDIA RTX CUDA 
run() --> train or test 

---

#### For Experience replay we'll use DEQUE data structure :
- DEQUE = Double ended queue - FIFO. - Remove elements from front.
- Experience = {State, Action, Reward, NextState, Terminated}
- ADD New Experiences from end and REMOVE old experiences from front

#### Training parameters

- We'll train the model indefinitely to achieve maximum rewards i.e. stop training by keyboard interrupt CTRL+C ,
- OR Until a specific Reward threshold is achieved.

#### Values in tensors;
- State, Actions, NextState, Reward
- these values will be fed to neural network

#### Target Network and Policy Network:
- Optimizer --> target q-value --> loss --> back propagation.
- optimize()--> 
- 1. train dqn from experience replay
- 2. compute target Q-val (y_true) using target_dqn
- 3. calc y_pred from policy_dqn
- 4. compute loss and backpropagate