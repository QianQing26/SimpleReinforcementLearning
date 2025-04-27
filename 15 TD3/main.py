import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from collections import deque
import random

from TD3 import TD3Agent, Actor, Critic


def train_agent(env, agent, num_episodes=500, max_steps=1000):
    rewards_history = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done, truncated = False, False
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
            print(
                f"Episode {episode+1}: Average reward over last 10 episodes: {avg_reward:.2f}"
            )
    return rewards_history


def test_agent(env, agent, num_episodes=5, max_steps=1000):
    rewards_history = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done, truncated = False, False
        steps = 0

        while not done and not truncated and steps < max_steps:
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            )
            action = agent.actor(state_tensor).cpu().detach().numpy()[0]
            scaled_action = action * env.action_space.high[0]
            next_state, reward, done, truncated, _ = env.step(scaled_action)

            episode_reward += reward
            state = next_state
            steps += 1

        rewards_history.append(episode_reward)
        print(f"Test Episode {episode+1}: Reward: {episode_reward:.2f}")
    return rewards_history


def plot_reward(rewards_history, title="Reward over Episodes"):
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.show()


def save_model(agent, path_actor, path_critic):
    os.makedirs(os.path.dirname(path_actor), exist_ok=True)
    torch.save(agent.actor.state_dict(), path_actor)
    torch.save(agent.critic.state_dict(), path_critic)


def load_model(
    path_actor, path_critic, input_dim, output_dim, max_action, hidden_dim=[400, 300]
):
    actor = Actor(input_dim, output_dim, hidden_dim, max_action)
    critic = Critic(input_dim, output_dim, hidden_dim)
    actor.load_state_dict(torch.load(path_actor))
    critic.load_state_dict(torch.load(path_critic))
    actor.eval()
    critic.eval()
    return actor, critic


def main():
    env = gym.make("Pendulum-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = TD3Agent(
        input_dim,
        output_dim,
        max_action,
        hidden_dim=[256, 256],
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
    )

    print("Training...")
    rewards = train_agent(env, agent, num_episodes=500)

    plot_reward(rewards)

    save_model(agent, "models/td3_actor.pth", "models/td3_critic.pth")

    env.close()

    env = gym.make("Pendulum-v1", render_mode="human")
    print("Testing...")
    test_agent(env, agent, num_episodes=5)
    env.close()


if __name__ == "__main__":
    main()
