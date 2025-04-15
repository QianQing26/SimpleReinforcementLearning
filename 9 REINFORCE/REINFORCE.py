import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from collections import namedtuple


class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[128]):
        super(Network, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.layers(x), dim=-1)


SavedAction = namedtuple("SavedAction", ["log_prob", "reward"])


class ReinforceAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=[128], lr=2e-3, gamma=0.99):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.policy = Network(input_dim, output_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_actions = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), None))
        return action.item()

    def update(self, rewards):
        R = 0
        policy_loss = []
        returns = []

        for r in reversed([sa.reward for sa in self.saved_actions]):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        for (log_prob, _), R in zip(self.saved_actions, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.saved_actions.clear()
