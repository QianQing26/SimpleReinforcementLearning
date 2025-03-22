import gymnasium as gym
import numpy as np
import pickle
from Qlearning import QLearningAgent


def train_agent(env, agent, episodes, save_path=None):
    """_summary_

    Args:
        env (_type_): _description_
        agent (_type_): _description_
        episodes (_type_): _description_
        save_path (_type_, optional): a string , "xxx.pkl" file
    """
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            agent.update(state, action, reward, next_state, terminated, truncated)
            state = next_state
        if (episode + 1) % 50 == 0:
            print(
                f"Episode: {episode+1},  Total reward: {total_reward},  Epsilon: {agent.epsilon}"
            )
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(agent, f)
        print(f"Trained Agent has been saved to {save_path}")


def test_agent(env, agent, episodes=20, max_steps=50):
    successes = 0
    total_steps = 0
    for episode in range(episodes):
        state, _ = env.reset()
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and steps <= 50:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            steps += 1

            if terminated and reward == 1:
                successes += 1

        total_steps += steps

    success_rate = successes / episodes
    avg_steps = total_steps / episodes
    print(f"Success Rate: {success_rate}  Average Step: {avg_steps}")
    return success_rate, avg_steps


if __name__ == "__main__":

    np.random.seed(42)

    env_name = "FrozenLake8x8-v1"
    env = gym.make(env_name, is_slippery=False)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = QLearningAgent(state_size, action_size)
    print("Training......")
    train_agent(
        env,
        agent,
        episodes=40000,
        save_path="agent/Qlearning_agent_FrozenLake8x8-v1.pkl",
    )
    print("\nTesting")
    test_agent(env, agent, episodes=10, max_steps=50)

    env.close()
