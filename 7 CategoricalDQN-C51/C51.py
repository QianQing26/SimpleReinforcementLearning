import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import threading
from collections import deque


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
            # return states, actions, rewards, next_states, dones
            return {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "dones": dones,
            }

    def __len__(self):
        return len(self.buffer)


class CategoricalQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, atom_size, support, hidden_dim=[128]):
        super(CategoricalQNetwork, self).__init__()
        self.output_dim = output_dim
        self.atom_size = atom_size
        self.support = support

        layer = []
        prev_size = input_dim
        for size in hidden_dim:
            layer.append(nn.Linear(prev_size, size))
            layer.append(nn.ReLU())
            prev_size = size
        self.feature_layer = nn.Sequential(*layer)
        self.value_layer = nn.Linear(prev_size, output_dim * atom_size)

    def forward(self, state):
        dist = self.feature_layer(state)
        dist = self.value_layer(dist)
        dist = dist.view(-1, self.output_dim, self.atom_size)
        dist = F.softmax(dist, dim=2)
        return dist

    def get_q_values(self, state):
        dist = self.forward(state)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values


class C51Agent:
    def __init__(
        self,
        env,
        gamma=0.99,
        epsilon=1.0,
        epsilon_end=0.005,
        epsilon_decay=0.998,
        lr=0.0005,
        atom_size=51,
        v_min=-10,
        v_max=10,
        batch_size=64,
        buffer_capacity=100000,
        hidden_layers=[128],
        update_target_freq=400,
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
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atom_size).to(self.device)
        self.delta_z = (v_max - v_min) / (atom_size - 1)

        self.main_net = CategoricalQNetwork(
            self.state_dim, self.action_dim, atom_size, self.support, hidden_layers
        ).to(self.device)
        self.target_net = CategoricalQNetwork(
            self.state_dim, self.action_dim, atom_size, self.support, hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_capacity)

        self.update_target_freq = update_target_freq
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_net.get_q_values(state)
            return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        samples = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(samples["states"]).to(self.device)
        actions = torch.LongTensor(samples["actions"]).to(self.device)
        rewards = torch.FloatTensor(samples["rewards"]).to(self.device)
        next_states = torch.FloatTensor(samples["next_states"]).to(self.device)
        dones = torch.FloatTensor(samples["dones"]).to(self.device)

        with torch.no_grad():
            next_dist = self.target_net(
                next_states
            )  # [batch_size, action_dim, atom_size]
            next_q = torch.sum(next_dist * self.support, dim=2)
            next_actions = next_q.argmax(dim=1)
            next_dist = next_dist[
                range(self.batch_size), next_actions
            ]  # [batch_size, atom_size]

            target_z = rewards.unsqueeze(1) + self.gamma * (
                1 - dones.unsqueeze(1)
            ) * self.support.unsqueeze(0)
            target_z = target_z.clamp(min=self.v_min, max=self.v_max)

            b = (target_z - self.v_min) / self.delta_z

            lower_idx = b.floor().long()
            upper_idx = b.ceil().long()

            projected_dist = torch.zeros_like(next_dist)

            # for i in range(self.batch_size):
            #     for j in range(self.atom_size):
            #         l = lower_idx[i][j]
            #         u = upper_idx[i][j]
            #         weight = next_dist[i][j]
            #         if l == u:
            #             projected_dist[i][l] += weight
            #         else:
            #             projected_dist[i][l] += weight * (u - b[i][j])
            #             projected_dist[i][u] += weight * (b[i][j] - l)

            # vectorized implementation
            # batch_idx = torch.arange(self.batch_size)
            batch_idx = (
                torch.arange(self.batch_size)
                .unsqueeze(1)
                .expand(-1, self.atom_size)
                .to(self.device)
            )  # [B, atom_size]
            eq_mask = lower_idx == upper_idx
            projected_dist[
                batch_idx.expand_as(lower_idx)[eq_mask], lower_idx[eq_mask]
            ] += next_dist[eq_mask]

            ne_mask = lower_idx != upper_idx
            l_idx = lower_idx[ne_mask]
            u_idx = upper_idx[ne_mask]
            b_val = b[ne_mask]
            weight = next_dist[ne_mask]
            batch_idx_expand = batch_idx.expand_as(lower_idx)[ne_mask]

            projected_dist[batch_idx_expand, l_idx] += weight * (u_idx.float() - b_val)
            projected_dist[batch_idx_expand, u_idx] += weight * (b_val - l_idx.float())

        dist = self.main_net(states)
        log_p = torch.log(dist[range(self.batch_size), actions])
        loss = -torch.sum(log_p * projected_dist, dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def push(self, *args):
        self.memory.push(*args)
