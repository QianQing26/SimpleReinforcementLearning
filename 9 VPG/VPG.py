import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np


# 定义网络结构
class VPGNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[128]):
        super(VPGNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 定义智能体
class VPGAgent:
    def __init__(self, input_dim, output_dim, lr=0.01, gamma=0.99, hidden_dim=[128]):
        self.network = VPGNetwork(input_dim, output_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.network(state)
        m = torch.distributions.Categorical(logits=probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action).unsqueeze(0))
        return action.item()

    def update(self):
        # 向量化计算 returns
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 向量化计算 policy_loss
        log_probs = torch.cat(self.log_probs)
        policy_loss = -torch.sum(log_probs * returns)

        # 更新网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.log_probs.clear()
        self.rewards.clear()
