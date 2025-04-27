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

from A3C import PolicyNetwork, ValueNetwork, worker


def train_agent(
    env_name, input_dim, output_dim, num_workers=4, num_episodes=1000, max_steps=1000
):
    shared_policy_net = PolicyNetwork(input_dim, output_dim, hidden_dim=[32, 32])
    shared_value_net = ValueNetwork(input_dim, hidden_dim=[32, 32])
    shared_policy_net.share_memory()
    shared_value_net.share_memory()

    optimizer = optim.Adam(
        list(shared_policy_net.parameters()) + list(shared_value_net.parameters()),
        lr=1e-3,
    )

    counter = mp.Value("i", 0)
    lock = mp.Lock()
    manager = mp.Manager()
    rewards_list = manager.list()

    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
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
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return list(rewards_list), shared_policy_net, shared_value_net


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
    env_name = "CartPole-v1"
    dummy_env = gym.make(env_name)
    input_dim = dummy_env.observation_space.shape[0]
    output_dim = dummy_env.action_space.n
    dummy_env.close()

    print("Training...")
    rewards, policy_net, value_net = train_agent(
        env_name, input_dim, output_dim, num_workers=4, num_episodes=1000
    )

    plot_reward(rewards)

    save_model(policy_net, value_net, "models/a3c_policy.pth", "models/a3c_value.pth")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
