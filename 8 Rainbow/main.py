import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from Rainbow import RainbowAgent  # 使用我们刚写好的RainbowAgent


def train_dqn(env, agent, num_episodes=1000, max_steps=200):
    rewards = []
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.push(state, action, reward, next_state, terminated or truncated)
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
    return rewards


def plot_rewards(rewards, title="Training rewards", window=15):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Rewards")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(rewards)), smoothed, label=f"{window}-ep Moving Avg"
        )
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def save_agent(agent, name="rainbow_cartpole"):
    torch.save(agent.main_net.state_dict(), f"{name}_main.pth")
    torch.save(agent.target_net.state_dict(), f"{name}_target.pth")
    print(f"Saved model weights to {name}_main.pth and {name}_target.pth")


def load_agent(agent, name="rainbow_cartpole"):
    agent.main_net.load_state_dict(torch.load(f"{name}_main.pth"))
    agent.target_net.load_state_dict(torch.load(f"{name}_target.pth"))
    agent.target_net.eval()
    print(f"Loaded model weights from {name}_main.pth and {name}_target.pth")
    return agent


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = RainbowAgent(
        env,
        update_target_freq=100,
        hidden_layers=[256],
        batch_size=128,
        v_min=0,
        v_max=300,
        beta_frames=50000,
        lr=6e-5,
    )

    train_rewards = train_dqn(env, agent, num_episodes=500, max_steps=300)
    plot_rewards(train_rewards, title="Rainbow DQN Training rewards")

    save_agent(agent, name="rainbow_cartpole")
    env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    load_agent(agent, name="rainbow_cartpole")
    test_rewards = test_dqn(env, agent, num_episodes=5)
    plot_rewards(test_rewards, title="Rainbow DQN Testing rewards")
    env.close()
