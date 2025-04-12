import numpy as np
import random
import threading
from collections import deque


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left]:
                    parent_idx = left
                else:
                    value -= self.tree[left]
                    parent_idx = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(
        self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, epsilon=1e-5
    ):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        self.lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1.0
        with self.lock:
            self.tree.add(max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / batch_size
        self.beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames,
        )
        self.frame += 1

        with self.lock:
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                value = random.uniform(a, b)
                idx, priority, data = self.tree.get_leaf(value)
                batch.append(data)
                idxs.append(idx)
                priorities.append(priority)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "idxs": idxs,
            "weights": is_weights.astype(np.float32),
        }

    def update_priorities(self, idxs, priorities):
        with self.lock:
            for idx, priority in zip(idxs, priorities):
                self.tree.update(idx, np.power(priority + self.epsilon, self.alpha))

    def __len__(self):
        return min(self.tree.data_pointer, self.tree.capacity)
