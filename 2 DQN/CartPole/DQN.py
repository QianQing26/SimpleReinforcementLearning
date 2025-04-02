import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import threading


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        with self.lock:
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
            return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[]):
        super(QNetwork, self).__init__()
        if not hidden_layers:
            self.fc = nn.Sequential(nn.Linear(input_dim, output_dim))
        else:
            Layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU()]
            for i in range(len(hidden_layers)):
                if i == len(hidden_layers) - 1:
                    Layers.append(nn.Linear(hidden_layers[i], output_dim))
                else:
                    Layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                    Layers.append(nn.ReLU())
            self.fc = nn.Sequential(*Layers)

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        epsilon=1.0,
        epsilon_end=0.005,
        epsilon_decay=0.9995,
        hidden_layers=[64, 64],
        lr=0.001,
        buffer_capacity=10000,
        batch_size=64,
        update_target_freq=100,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.main_net = QNetwork(self.state_dim, self.action_dim, hidden_layers).to(
            self.device
        )
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden_layers).to(
            self.device
        )
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

        self.steps_done = 0
        self.update_target_frequency = update_target_freq

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.main_net(state)
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

        q_values = self.main_net(states)
        next_q_values = self.target_net(next_states).detach()
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
        self.steps_done += 1

        if self.steps_done % self.update_target_frequency == 0:
            self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.main_net.state_dict())


if __name__ == "__main__":
    # 测试QNetwork是否正确
    input_dim = 4
    output_dim = 2
    hidden_layers = [8, 4]
    net = QNetwork(input_dim, output_dim, hidden_layers)
    print(net)
