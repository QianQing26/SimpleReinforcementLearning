import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from collections import deque
import os


# Policy Network (Gaussian Policy)
class Actor(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=[256, 256],
        max_action=1.0,
        log_std_min=-20,
        log_std_max=2,
    ):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim[1], output_dim)
        self.log_std_layer = nn.Linear(hidden_dim[1], output_dim)

        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        # 计算log_prob
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


# Two Q Networks
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=[256, 256]):
        super(Critic, self).__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 1),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 1),
        )

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        q1 = self.q1_net(xu)
        q2 = self.q2_net(xu)
        return q1, q2


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# SAC Agent
class SACAgent:
    def __init__(
        self,
        input_dim,
        action_dim,
        max_action,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=1000000,
        batch_size=256,
        automatic_entropy_tuning=True,
        target_entropy=None,
    ):
        self.device = "cpu"

        self.actor = Actor(input_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(input_dim, action_dim).to(self.device)
        self.critic_target = Critic(input_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action

        # entropy部分
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = 0.2

    def select_action(self, state, eval_mode=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if eval_mode:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.cpu().detach().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().detach().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha() * next_log_prob
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        new_action, log_prob = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_action)
        actor_loss = (self.alpha() * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update target critic
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        else:
            return torch.tensor(self.alpha).to(self.device)

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
