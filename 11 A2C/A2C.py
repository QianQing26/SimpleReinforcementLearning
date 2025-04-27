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


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=[128]):
        super(ValueNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(-1)


SavedAction = namedtuple("SavedAction", ["log_prob", "state", "reward"])


class A2CAgent:
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=[128],
        policy_lr=1e-3,
        value_lr=1e-3,
        gamma=0.99,
    ):
        # self.device = "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_net = PolicyNetwork(input_dim, output_dim, hidden_dim).to(
            self.device
        )
        self.value_net = ValueNetwork(input_dim, hidden_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.gamma = gamma
        self.saved_actions = []

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state, None))
        return action.item()

    def update(self):
        R = 0
        returns = []

        for r in reversed([sa.reward for sa in self.saved_actions]):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).float().to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        policy_losses = []
        value_losses = []

        for saved_action, R in zip(self.saved_actions, returns):
            log_prob, state, _ = saved_action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            value = self.value_net(state_tensor)
            advantage = R - value

            policy_losses.append(-log_prob * advantage.detach())
            value_losses.append(F.mse_loss(value, R))

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        policy_loss = torch.cat(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()

        policy_loss.backward()
        value_loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

        self.saved_actions.clear()
