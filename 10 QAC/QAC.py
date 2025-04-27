import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from tqdm import tqdm


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[128]):
        super(PolicyNetwork, self).__init__()
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


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[128]):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


SavedAction = namedtuple("SavedAction", ["log_prob", "state", "action", "reward"])


class QACAgent:
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=[128],
        policy_lr=1e-3,
        q_lr=1e-3,
        gamma=0.99,
    ):
        self.device = "cpu"
        self.policy_net = PolicyNetwork(input_dim, output_dim, hidden_dim).to(
            self.device
        )
        self.q_net = QNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.gamma = gamma
        self.saved_actions = []

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_actions.append(
            SavedAction(m.log_prob(action), state, action.item(), None)
        )
        return action.item()

    def update(self):
        R = 0
        policy_loss = []
        q_loss = []
        returns = []

        for r in reversed([sa.reward for sa in self.saved_actions]):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).float().to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        for saved_action, R in zip(self.saved_actions, returns):
            log_prob, state, action, _ = saved_action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            q_value = q_values[0, action]

            policy_loss.append(-log_prob * q_value.detach())
            q_loss.append(F.mse_loss(q_value, R))

        self.policy_optimizer.zero_grad()
        self.q_optimizer.zero_grad()

        policy_loss = torch.cat(policy_loss).sum()
        q_loss = torch.stack(q_loss).sum()

        policy_loss.backward()
        q_loss.backward()

        self.policy_optimizer.step()
        self.q_optimizer.step()

        self.saved_actions.clear()
