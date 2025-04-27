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

from SAC import SACAgent


def train_agent(env, agent, num_episodes=500, max_steps=1000):
    rewards_history = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done, truncated = False, False
        episode_reward = 0
        steps = 0

        while not done and not truncated and steps < max_steps:
            action = agent.select_action(state)
            scaled_action = action * env.action_space.high[0]
            next_state, reward, done, truncated, _ = env.step(scaled_action)

            agent.push_transition(state, action, reward, next_state, done)
            agent.update()

            episode_reward += reward
            state = next_state
            steps += 1

        rewards_history.append(episode_reward)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}: Average Reward (last 10): {avg_reward:.2f}")

    return rewards_history


def test_agent(env, agent, num_episodes=5, max_steps=1000):
    for episode in range(num_episodes):
        state, _ = env.reset()
        done, truncated = False, False
        episode_reward = 0
        steps = 0

        while not done and not truncated and steps < max_steps:
            action = agent.select_action(state, eval_mode=True)
            scaled_action = action * env.action_space.high[0]
            next_state, reward, done, truncated, _ = env.step(scaled_action)

            episode_reward += reward
            state = next_state
            steps += 1

        print(f"Test Episode {episode+1}: Reward {episode_reward:.2f}")


def plot_reward(rewards_history, title="Reward over Episodes"):
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    env = gym.make("Pendulum-v1")
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = SACAgent(input_dim, action_dim, max_action)

    print("Training...")
    rewards = train_agent(env, agent, num_episodes=500)

    plot_reward(rewards)

    env.close()

    env = gym.make("Pendulum-v1", render_mode="human")
    print("Testing...")
    test_agent(env, agent, num_episodes=5)
    env.close()


if __name__ == "__main__":
    main()
