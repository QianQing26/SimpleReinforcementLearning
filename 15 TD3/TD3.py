import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from collections import deque
import random


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[400, 300], max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        layers = []
        prev_dim = input_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.max_action * torch.tanh(self.layers(x))


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=[400, 300]):
        super(Critic, self).__init__()
        # Q1网络
        self.q1_fc1 = nn.Linear(input_dim + action_dim, hidden_dim[0])
        self.q1_fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.q1_out = nn.Linear(hidden_dim[1], 1)
        # Q2网络
        self.q2_fc1 = nn.Linear(input_dim + action_dim, hidden_dim[0])
        self.q2_fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.q2_out = nn.Linear(hidden_dim[1], 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_fc1(xu))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(xu))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

    def q1(self, state, action):
        xu = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(xu))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        return q1


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
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class TD3Agent:
    def __init__(
        self,
        input_dim,
        output_dim,
        max_action,
        hidden_dim=[400, 300],
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_capacity=1000000,
        batch_size=256,
    ):
        self.device = "cpu"

        self.actor = Actor(input_dim, output_dim, hidden_dim, max_action).to(
            self.device
        )
        self.actor_target = Actor(input_dim, output_dim, hidden_dim, max_action).to(
            self.device
        )
        self.critic = Critic(input_dim, output_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(input_dim, output_dim, hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.total_it = 0  # 更新步数

    def select_action(self, state, noise_scale=0.1):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().detach().numpy()[0]
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
