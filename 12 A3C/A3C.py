import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
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
        self.shared_layers = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.shared_layers(x)
        probs = F.softmax(self.policy_head(x), dim=-1)
        return probs


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=[128]):
        super(ValueNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.shared_layers = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        value = self.value_head(x).squeeze(-1)
        return value


SavedAction = namedtuple("SavedAction", ["log_prob", "state", "reward"])


class A3CAgent:
    def __init__(
        self,
        input_dim,
        output_dim,
        shared_policy_net,
        shared_value_net,
        optimizer,
        gamma=0.99,
    ):
        self.device = "cpu"
        self.policy_net = shared_policy_net
        self.value_net = shared_value_net
        self.optimizer = optimizer
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
        if not self.saved_actions:
            return
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

        loss = torch.cat(policy_losses).sum() + torch.stack(value_losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.saved_actions.clear()


def worker(
    worker_id,
    env_name,
    shared_policy_net,
    shared_value_net,
    optimizer,
    counter,
    lock,
    num_episodes,
    max_steps,
    rewards_list,
):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = A3CAgent(
        input_dim, output_dim, shared_policy_net, shared_value_net, optimizer
    )

    local_rewards = []
    while True:
        with lock:
            if counter.value >= num_episodes:
                break
            counter.value += 1
            episode_idx = counter.value

        state, _ = env.reset()
        episode_reward = 0
        done, truncated = False, False
        steps = 0
        while not done and not truncated and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.saved_actions[-1] = agent.saved_actions[-1]._replace(reward=reward)
            episode_reward += reward
            steps += 1
            state = next_state

        agent.update()
        local_rewards.append(episode_reward)

        if episode_idx % 50 == 0:
            avg_reward = np.mean(local_rewards[-50:])
            print(
                f"Worker {worker_id} - Episode {episode_idx}: Average reward over last 50: {avg_reward:.2f}"
            )

    rewards_list += local_rewards
    env.close()
