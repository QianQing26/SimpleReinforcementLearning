import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import threading


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)  # 使用 deque 并设置最大长度
        self.lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        with self.lock:
            self.buffer.append(
                (state, action, reward, next_state, done)
            )  # 直接追加到 deque

    def sample(self, batch_size):
        with self.lock:
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
            return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingNoisyQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[128]):
        super(DuelingNoisyQNetwork, self).__init__()
        # 共享网络层
        layers = []
        prev_size = input_dim
        for size in hidden_layers:
            layers.append(NoisyLinear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        self.shared_net = nn.Sequential(*layers)

        self.value_stream = nn.Sequential(
            NoisyLinear(prev_size, 128), nn.ReLU(), NoisyLinear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(prev_size, 128), nn.ReLU(), NoisyLinear(128, output_dim)
        )

    def forward(self, x):
        shared = self.shared_net(x)
        value = self.value_stream(shared)
        advantages = self.advantage_stream(shared)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class NoisyDuelDQNAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        hidden_layers=[128],
        lr=0.001,
        buffer_capacity=10000,
        batch_size=64,
        update_target_freq=600,
    ):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.main_net = DuelingNoisyQNetwork(
            self.state_dim, self.action_dim, hidden_layers
        ).to(self.device)
        self.target_net = DuelingNoisyQNetwork(
            self.state_dim, self.action_dim, hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

        self.steps_done = 0
        self.update_target_freq = update_target_freq

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_net(state)
            self.main_net.reset_noise()
            return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.main_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_actions = self.main_net(next_states).argmax(1)
            next_q = (
                self.target_net(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze()
            )
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.main_net.reset_noise()
        self.target_net.reset_noise()

        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
