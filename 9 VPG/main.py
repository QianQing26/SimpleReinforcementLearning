import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from VPG import VPGAgent


def train_agent(env, agent, num_episodes=1000):
    rewards_history = []
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        truncated = False
        rewards = 0.0
        while not (done or truncated):
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            rewards += reward
        agent.update()
        rewards_history.append(rewards)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {rewards}")
    return rewards_history


def test_agent(env, agent, num_episodes=10):
    rewards = 0.0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        while not (done or truncated):
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards += episode_reward
    print(f"Average Reward over {num_episodes} episodes: {rewards / num_episodes}")


def plot_reward(rewards_history, title):
    plt.plot(rewards_history)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()


# 主程序
if __name__ == "__main__":
    env = gym.make("CartPole-v1", max_episode_steps=1000)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = VPGAgent(input_dim, output_dim)

    rewards_history = train_agent(env, agent, num_episodes=500)
    plot_reward(rewards_history, "VPG Training Reward")

    env.close()
    env = gym.make("CartPole-v1", max_episode_steps=1000, render_mode="human")
    test_agent(env, agent)

    env.close()
