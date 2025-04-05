import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import threading


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        with self.lock:
            max_priority = self.priorities.max() if self.buffer else 1.0
            if len(self.buffer) < self.capacity:
                self.buffer.append((state, action, reward, next_state, done))
            else:
                self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        with self.lock:
            priorities = self.priorities[: len(self.buffer)]
            probs = priorities**self.alpha
            probs /= probs.sum()

            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]

            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-self.beta)
            weights /= weights.max()

            states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))
            return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        with self.lock:
            for idx, priority in zip(batch_indices, batch_priorities):
                self.priorities[idx] = max(priority, 1e-6)  # Prevent zero priority

    def __len__(self):
        return len(self.buffer)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=[128]):
        super(DuelingQNetwork, self).__init__()
        layers = []
        prev_size = input_dim
        for size in hidden_size:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        self.shared = nn.Sequential(*layers)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.shared(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DuelDQNAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        epsilon=1.0,
        epsilon_end=0.005,
        epsilon_decay=0.9995,
        hidden_layers=[128],
        lr=0.0001,
        buffer_capacity=100000,
        batch_size=64,
        update_target_freq=300,
        alpha=0.6,
        beta=0.4,
        beta_annealing=0.001,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.main_net = DuelingQNetwork(
            self.state_dim, self.action_dim, hidden_layers
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            self.state_dim, self.action_dim, hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(
            buffer_capacity, alpha, beta, beta_annealing
        )

        self.steps_done = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_net(state)
        return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size)
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q = self.main_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.main_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        td_errors = (current_q - target_q).abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors + 1e-6)

        loss = (
            weights * F.smooth_l1_loss(current_q, target_q, reduction="none")
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
