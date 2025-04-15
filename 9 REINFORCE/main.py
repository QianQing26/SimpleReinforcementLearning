from REINFORCE import ReinforceAgent, SavedAction, Network
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def train_agent(env, agent, num_episodes=1000, max_steps=1000):
    rewards_history = []
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        # state, _ = env.reset(seed=123, options={"low": -0.7, "high": 0.5})
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

        agent.update(rewards=episode_reward)
        rewards_history.append(episode_reward)
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(
                f"Episode {episode+1}: Average reward over last 50 episodes: {avg_reward:.2f}"
            )
    return rewards_history


def test_agent(env, agent, num_episodes=10, max_steps=1000):
    agent.policy.eval()
    rewards_history = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        # state, _ = env.reset(seed=123, options={"low": -0.7, "high": 0.5})
        episode_reward = 0
        done, truncated = False, False
        steps = 0
        while not done and not truncated and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            # agent.saved_actions[-1] = agent.saved_actions[-1]._replace(reward=reward)
            episode_reward += reward
            steps += 1
            state = next_state

        # agent.update(rewards=episode_reward)
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


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path, input_dim, output_dim, hidden_dim=128):
    model = Network(input_dim, output_dim, hidden_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def main():
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = ReinforceAgent(
        input_dim, output_dim, hidden_dim=[32, 32], lr=5e-3, gamma=0.99
    )

    print("Training...")
    rewards = train_agent(env, agent, num_episodes=1000)

    plot_reward(rewards)

    save_model(agent.policy, "models/reinforce_cartpole.pth")
    env.close()
    env = gym.make("CartPole-v1", render_mode="human")
    print("Testing...")
    test_agent(env, agent, num_episodes=5)

    env.close()


if __name__ == "__main__":
    main()
