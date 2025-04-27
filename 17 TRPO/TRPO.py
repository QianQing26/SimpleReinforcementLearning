import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import scipy.optimize
import copy


# Policy Network (Gaussian)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[64, 64]):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim[1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable log_std

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action

    def evaluate_actions(self, states, actions):
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_probs, entropy


# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=[64, 64]):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], 1),
        )

    def forward(self, state):
        return self.net(state)


# Helper functions for TRPO
def flat_params(model):
    return torch.cat([param.view(-1) for param in model.parameters()])


def set_flat_params(model, flat_params):
    idx = 0
    for param in model.parameters():
        n_param = param.numel()
        param.data.copy_(flat_params[idx : idx + n_param].view(param.size()))
        idx += n_param


def flat_grad(output, inputs, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(
        output, inputs, retain_graph=retain_graph, create_graph=create_graph
    )
    return torch.cat([grad.view(-1) for grad in grads])


# Conjugate Gradient Method
def conjugate_gradient(Avp_func, b, cg_iters=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        Avp = Avp_func(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x


# TRPO Agent
class TRPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_kl=1e-2,
        damping=1e-2,
        gamma=0.99,
        gae_lambda=0.97,
        vf_lr=1e-3,
    ):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_fn = ValueNetwork(state_dim)
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=vf_lr)
        self.max_kl = max_kl
        self.damping = damping
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.policy.get_action(state)
        return action.detach().numpy()[0]

    def compute_gae(self, rewards, masks, values, next_value):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * masks[step]
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, trajectories):
        states = torch.tensor(np.array(trajectories["states"]), dtype=torch.float32)
        actions = torch.tensor(np.array(trajectories["actions"]), dtype=torch.float32)
        rewards = trajectories["rewards"]
        masks = trajectories["masks"]

        with torch.no_grad():
            values = self.value_fn(states).squeeze().tolist()
        next_state = torch.tensor(
            np.array(trajectories["next_states"][-1]), dtype=torch.float32
        )
        next_value = self.value_fn(next_state.unsqueeze(0)).item()

        returns = self.compute_gae(rewards, masks, values, next_value)

        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - self.value_fn(states).squeeze()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Old log probs
        old_log_probs, _ = self.policy.evaluate_actions(states, actions)
        old_log_probs = old_log_probs.detach()

        # Surrogate loss
        def surrogate_loss():
            new_log_probs, _ = self.policy.evaluate_actions(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            return -(ratio * advantages).mean()

        loss = surrogate_loss()

        # Compute policy gradient
        policy_grads = flat_grad(loss, self.policy.parameters(), retain_graph=True)

        # Define Hessian-vector product function
        def hessian_vector_product(vector):
            new_log_probs, _ = self.policy.evaluate_actions(states, actions)
            kl = (old_log_probs - new_log_probs).mean()
            grads = flat_grad(kl, self.policy.parameters(), create_graph=True)
            kl_v = (grads * vector).sum()
            grads2 = flat_grad(kl_v, self.policy.parameters(), retain_graph=True)
            return grads2 + self.damping * vector

        step_dir = conjugate_gradient(hessian_vector_product, policy_grads)
        shs = 0.5 * (step_dir * hessian_vector_product(step_dir)).sum(0, keepdim=True)
        step_size = torch.sqrt(self.max_kl / (shs + 1e-8))
        full_step = step_dir * step_size

        old_params = flat_params(self.policy)

        def set_and_eval(step):
            new_params = old_params + step
            set_flat_params(self.policy, new_params)
            return surrogate_loss()

        # Line search
        for step_frac in [0.5**i for i in range(10)]:
            loss_new = set_and_eval(full_step * step_frac)
            if loss_new < loss:
                break
        else:
            set_flat_params(self.policy, old_params)

        # Update Value Function
        value_loss = F.mse_loss(self.value_fn(states).squeeze(), returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
