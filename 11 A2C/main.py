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

from A2C import A2CAgent, PolicyNetwork, ValueNetwork


def train_agent(env, agent, num_episodes=1000, max_steps=1000):
    rewards_history = []
    for episode in tqdm(range(num_episodes)):
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
        rewards_history.append(episode_reward)
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(
                f"Episode {episode+1}: Average reward over last 50 episodes: {avg_reward:.2f}"
            )
    return rewards_history


def test_agent(env, agent, num_episodes=10, max_steps=1000):
    agent.policy_net.eval()
    rewards_history = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done, truncated = False, False
        steps = 0
        while not done and not truncated and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            state = next_state

        rewards_history.append(episode_reward)
        print(f"Test episode {episode+1}: Total reward: {episode_reward:.2f}")
    return rewards_history


def plot_reward(rewards_history, title="Reward over Episodes"):
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.show()


def save_model(policy_model, value_model, path_policy, path_value):
    os.makedirs(os.path.dirname(path_policy), exist_ok=True)
    torch.save(policy_model.state_dict(), path_policy)
    torch.save(value_model.state_dict(), path_value)


def load_model(path_policy, path_value, input_dim, output_dim, hidden_dim=[128]):
    policy_model = PolicyNetwork(input_dim, output_dim, hidden_dim)
    value_model = ValueNetwork(input_dim, hidden_dim)
    policy_model.load_state_dict(torch.load(path_policy))
    value_model.load_state_dict(torch.load(path_value))
    policy_model.eval()
    value_model.eval()
    return policy_model, value_model


def main():
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = A2CAgent(
        input_dim,
        output_dim,
        hidden_dim=[32, 32],
        policy_lr=5e-3,
        value_lr=5e-3,
        gamma=0.99,
    )

    print("Training...")
    rewards = train_agent(env, agent, num_episodes=1000)

    plot_reward(rewards)

    save_model(
        agent.policy_net,
        agent.value_net,
        "models/a2c_policy.pth",
        "models/a2c_value.pth",
    )
    env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    print("Testing...")
    test_agent(env, agent, num_episodes=5)
    env.close()


if __name__ == "__main__":
    main()
