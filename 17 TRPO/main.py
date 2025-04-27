import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import scipy.optimize
import copy

from TRPO import TRPOAgent


# Training and Testing Functions
def train(env, agent, num_episodes=500, max_steps=1000):
    all_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        trajectories = {
            "states": [],
            "actions": [],
            "rewards": [],
            "masks": [],
            "next_states": [],
        }
        episode_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state)
            scaled_action = action * env.action_space.high[0]
            next_state, reward, done, truncated, _ = env.step(scaled_action)

            trajectories["states"].append(state)
            trajectories["actions"].append(action)
            trajectories["rewards"].append(reward)
            trajectories["masks"].append(1 - done)
            trajectories["next_states"].append(next_state)

            episode_reward += reward
            state = next_state
            if done or truncated:
                break

        agent.update(trajectories)
        all_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Reward: {np.mean(all_rewards[-10:]):.2f}")

    return all_rewards


def test(env, agent, num_episodes=5, max_steps=1000):
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(state)
            scaled_action = action * env.action_space.high[0]
            next_state, reward, done, truncated, _ = env.step(scaled_action)
            episode_reward += reward
            state = next_state
            if done or truncated:
                break
        print(f"Test Episode {ep+1}: Reward {episode_reward:.2f}")


def plot_rewards(rewards, title="Rewards over Episodes"):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = TRPOAgent(state_dim, action_dim)

    print("Training...")
    rewards = train(env, agent, num_episodes=500)

    plot_rewards(rewards)

    env.close()

    env = gym.make("Pendulum-v1", render_mode="human")
    print("Testing...")
    test(env, agent, num_episodes=5)
    env.close()


if __name__ == "__main__":
    main()
