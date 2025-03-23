import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from DQN import DQNAgent


def train_dqn(env, agent, num_episodes=1000, max_steps=200, target_update_frequency=10):
    rewards = []
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, terminated)
            state = next_state
            episode_reward += reward
            agent.update()
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_net()
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode+1}/{num_episodes}   Reward: {episode_reward:.2f},  Epsilon: {agent.epsilon:.3f}"
            )
    return rewards


def test_dqn(env, agent, num_episodes=10, max_steps=200):
    rewards = []
    epsilon = agent.epsilon
    agent.epsilon = 0.0  # Disable exploration during testing
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}   Reward: {episode_reward:.2f}")
    agent.epsilon = epsilon
    return rewards


def plot_rewards(rewards, title="Training rewards"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Rewards")
    plt.plot(
        np.convolve(rewards, np.ones(20) / 20, mode="valid"), label="Smoothened rewards"
    )
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)
    train_rewards = train_dqn(
        env, agent, num_episodes=800, max_steps=250, target_update_frequency=6
    )
    plot_rewards(train_rewards, title="Training rewards")
    env.close()
    env = gym.make("CartPole-v1", render_mode="human")
    test_rewards = test_dqn(env, agent)
    plot_rewards(test_rewards, title="Testing rewards")
    env.close()

    # env = gym.make("CartPole-v1", render_mode="human")
    # env.reset()
    # for i in tqdm(range(50000)):
    #     action = env.action_space.sample()
    #     next_state, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         env.reset()
    # env.close()
