import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Qlearning import QLearningAgent


def train_agent(env, agent, episodes, save_path=None):
    """_summary_

    Args:
        env (_type_): _description_
        agent (_type_): _description_
        episodes (_type_): _description_
        save_path (_type_, optional): a string , "xxx.pkl" file
    """
    reward_history = []
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
            reward_history.append(total_reward)
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(agent, f)
        print(f"Trained Agent has been saved to {save_path}")
    return reward_history


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

            if terminated:
                successes += 1

        total_steps += steps

    success_rate = successes / episodes
    avg_steps = total_steps / episodes
    print(f"Success Rate: {success_rate}  Average Step: {avg_steps}")
    return success_rate, avg_steps


if __name__ == "__main__":

    np.random.seed(42)

    env_name = "Blackjack-v1"
    env = gym.make(env_name, natural=False, sab=False)

    state_size0, state_size1, state_size2 = (
        env.observation_space[0].n,
        env.observation_space[1].n,
        env.observation_space[2].n,
    )
    action_size = env.action_space.n

    agent = QLearningAgent(
        state_size0, action_size
    )  # initialize agent and modify the qtable later
    agent.set_q_table(np.zeros((state_size0, state_size1, state_size2, action_size)))
    print("Training......")
    reward_history = train_agent(
        env,
        agent,
        episodes=60000,
        save_path="agent/Qlearning_agent_Blackjack-v1.pkl",
    )
    print("\nTesting......")
    test_agent(env, agent, episodes=10, max_steps=50)

    env.close()

    # 先对reward_history进行平滑处理
    reward_history = np.array(reward_history)
    reward_history = np.convolve(reward_history, np.ones(50) / 50, mode="full")

    # 画出reward_history曲线
    plt.plot(reward_history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Agent on Blackjack-v1")
    plt.show()
