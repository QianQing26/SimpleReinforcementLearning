import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# Policy Network (Gaussian Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[64, 64]):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim[1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable log_std

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(axis=-1)

    def evaluate_actions(self, states, actions):
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_probs, entropy


# Value Network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=[64, 64]):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], 1),
        )

    def forward(self, state):
        return self.net(state)


# PPO Agent
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        policy_lr=3e-4,
        value_lr=1e-3,
        epochs=10,
        batch_size=64,
    ):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_fn = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=value_lr)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, log_prob = self.policy.get_action(state)
        return action.detach().numpy()[0], log_prob.detach().item()

    def compute_gae(self, rewards, masks, values, next_value):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * masks[step]
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, trajectories):
        states = torch.tensor(np.array(trajectories["states"]), dtype=torch.float32)
        actions = torch.tensor(np.array(trajectories["actions"]), dtype=torch.float32)
        old_log_probs = torch.tensor(
            np.array(trajectories["log_probs"]), dtype=torch.float32
        )
        rewards = trajectories["rewards"]
        masks = trajectories["masks"]

        with torch.no_grad():
            values = self.value_fn(states).squeeze().tolist()
        next_state = torch.tensor(
            np.array(trajectories["next_states"][-1]), dtype=torch.float32
        )
        next_value = self.value_fn(next_state.unsqueeze(0)).item()

        returns = self.compute_gae(rewards, masks, values, next_value)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - self.value_fn(states).squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(
            states, actions, old_log_probs, returns, advantages
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) in loader:
                # New log probs and entropy
                new_log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Value function loss
                value_loss = F.mse_loss(
                    self.value_fn(batch_states).squeeze(), batch_returns
                )

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
