import gymnasium as gym
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.99,
        epsilon_decay=0.9999,
        min_epsilon=0.001,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, terminated, truncated):
        current_q = self.q_table[state][action]
        max_feature_q = np.max(self.q_table[next_state])
        if terminated or truncated:
            max_feature_q = 0
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
        new_q = current_q - self.learning_rate * (
            current_q - (reward + self.discount_factor * max_feature_q)
        )
        self.q_table[state][action] = new_q

    def get_q_table(self):
        return self.q_table

    def set_q_table(self, q_table):
        self.q_table = q_table
