# SimpleReinforcementLearning
## Introduction
This is a simple implementation of Reinforcement Learning algorithms in Python. 

The goal of this repository is to provide clean, readable, and educational implementations for key RL algorithms.

The algorithms included are:

- **Q-Learning**

- **DQN (Deep Q-Network)**

- **DoubleDQN**

- **DuelDQN (Dueling DQN)**

- **NoisyDQN**

- **PrioritizedDQN (PER DQN)**

- **C51 (Categorical DQN)**

- **Rainbow DQN**

- **REINFORCE**

- **QAC (Q Actor-Critic)**

- **A2C (Advantage Actor-Critic)**

- **A3C (Asynchronous Advantage Actor-Critic)**

- **DDPG (Deep Deterministic Policy Gradient)**

- **TD3 (Twin Delayed DDPG)**

- **SAC (Soft Actor-Critic)**

- **TRPO (Trust Region Policy Optimization)**

- **PPO (Proximal Policy Optimization)**

## Requirements
- Python 3.13
- numpy==2.1.2
- torch==2.6.0
- tqdm==4.67.1
- matplotlib==3.10.1
- gymnasium==1.0.0
- scipy==1.15.2


## Project Features
- **🧹 Simple and Clean Code Structure** : Each algorithm is implemented clearly, with minimal but necessary abstraction — making it easy to follow and modify.
- **📚 Educational Purpose** : The code is optimized for readability and learning, not just performance.
- **🧩 Modular Components** : Networks (e.g., Policy Networks, Value Networks), Replay Buffers, and Agents are modularized for easier understanding.
- **🔥 Modern and Correct Implementations** : Many parts of the code, especially complex algorithms like Rainbow, SAC, TD3, PPO, TRPO, etc., are refined with the help of ChatGPT, ensuring they align with current best practices.
- **🚀 Ready-to-Train** : Each agent includes ```train```, ```test```, and ```plotting``` utilities to quickly visualize performance.
- **🔧 Flexible** : Easy to adjust hyper-parameters such as learning rates, hidden layer sizes, discount factors (```gamma```), and optimizers.

## How to Run
To run an algorithm, simply run the corresponding ```main.py```.

## Future Work
- **🎨 More Visualizations** : Add more visualizations to make the learning process more intuitive and easier to understand.
- **🐍 More Algorithms** : Implement more algorithms, such as A2C, A3C, and PPO.
- **🤖 More Environments** : Add more environments, such as Atari games, MuJoCo, and PyBullet.
- Include model-saving and checkpointing functionality.
- (Optional) Provide Colab notebooks for faster testing




