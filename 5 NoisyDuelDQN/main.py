import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from NoisyDuelDQN import NoisyDuelDQNAgent


def train_dqn(env, agent, num_episodes=1000, max_steps=200):
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
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}   Reward: {episode_reward:.2f}")
    return rewards


def test_dqn(env, agent, num_episodes=10, max_steps=200):
    rewards = []
    # epsilon = agent.epsilon
    # agent.epsilon = 0.0  # Disable exploration during testing
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
    # agent.epsilon = epsilon
    return rewards


def load_agent(agent, main_net_path, target_net_path):
    agent.main_net.load_state_dict(torch.load(main_net_path))
    agent.target_net.load_state_dict(torch.load(target_net_path))
    # agent.update_target_net()
    agent.target_net.eval()
    agent.epsilon = agent.epsilon_end
    print("Agent loaded successfully")
    return agent


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
    agent = NoisyDuelDQNAgent(env, update_target_freq=600)
    # agent = load_agent(
    #     agent,
    #     "3 DoubleDQN/CartPole/dqn_agent_main_net.pth",
    #     "3 DoubleDQN/CartPole/dqn_agent_target_net.pth",
    # )
    train_rewards = train_dqn(env, agent, num_episodes=500, max_steps=600)
    plot_rewards(train_rewards, title="Training rewards")
    env.close()
    # torch.save(
    #     agent.main_net.state_dict(), "3 DoubleDQN/CartPole/dqn_agent_main_net.pth"
    # )
    # torch.save(
    #     agent.target_net.state_dict(), "3 DoubleDQN/CartPole/dqn_agent_target_net.pth"
    # )
    env = gym.make("CartPole-v1", render_mode="human")
    test_rewards = test_dqn(env, agent)
    plot_rewards(test_rewards, title="Noisy Dueling DQN Testing rewards")
    env.close()
