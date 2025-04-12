import torch
import torch.nn as nn
import torch.nn.functional as F
from NoisyLinear import NoisyLinear
from PER import PrioritizedReplayBuffer


class RainbowCategoricalQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, atom_size, support, hidden_dim=[128]):
        super(RainbowCategoricalQNetwork, self).__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.support = support

        layers = []
        prev_dim = input_dim
        for hidden in hidden_dim:
            layers.append(NoisyLinear(prev_dim, hidden))
            layers.append(nn.ReLU())
            prev_dim = hidden
        self.feature = nn.Sequential(*layers)

        self.value_stream = nn.Sequential(
            NoisyLinear(prev_dim, prev_dim),
            nn.ReLU(),
            NoisyLinear(prev_dim, atom_size),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(prev_dim, prev_dim),
            nn.ReLU(),
            NoisyLinear(prev_dim, action_dim * atom_size),
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature(x)

        value = self.value_stream(features).view(batch_size, 1, self.atom_size)
        advantage = self.advantage_stream(features).view(
            batch_size, self.action_dim, self.atom_size
        )

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        return dist

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def get_q_values(self, x):
        dist = self.forward(x)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values


class RainbowAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        lr=0.0002,
        atom_size=51,
        v_min=-10,
        v_max=10,
        batch_size=64,
        buffer_capacity=100000,
        hidden_layers=[128],
        update_target_freq=360,
        beta_start=0.4,
        beta_frames=100000,
        alpha=0.6,
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atom_size).to(self.device)
        self.delta_z = (v_max - v_min) / (atom_size - 1)

        self.main_net = RainbowCategoricalQNetwork(
            self.state_dim, self.action_dim, atom_size, self.support, hidden_layers
        ).to(self.device)
        self.target_net = RainbowCategoricalQNetwork(
            self.state_dim, self.action_dim, atom_size, self.support, hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(buffer_capacity, alpha)
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target_freq = update_target_freq
        self.steps_done = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_net.get_q_values(state)
            return q_values.argmax(1).item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # beta = min(
        #     1.0,
        #     self.beta_start
        #     + self.steps_done * (1.0 - self.beta_start) / self.beta_frames,
        # )
        samples = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(samples["states"]).to(self.device)
        actions = torch.LongTensor(samples["actions"]).to(self.device)
        rewards = torch.FloatTensor(samples["rewards"]).to(self.device)
        next_states = torch.FloatTensor(samples["next_states"]).to(self.device)
        dones = torch.FloatTensor(samples["dones"]).to(self.device)
        weights = torch.FloatTensor(samples["weights"]).to(self.device)
        indices = samples["idxs"]

        # with torch.no_grad():
        #     next_dist = self.target_net(next_states)
        #     next_q = torch.sum(next_dist * self.support, dim=2)
        #     next_actions = next_q.argmax(dim=1)
        #     next_dist = next_dist[range(self.batch_size), next_actions]

        #     target_z = rewards.unsqueeze(1) + self.gamma * (
        #         1 - dones.unsqueeze(1)
        #     ) * self.support.unsqueeze(0)
        #     target_z = target_z.clamp(self.v_min, self.v_max)
        #     b = (target_z - self.v_min) / self.delta_z
        #     lower_idx = b.floor().long()
        #     upper_idx = b.ceil().long()

        #     projected_dist = torch.zeros_like(next_dist)
        #     eq_mask = upper_idx == lower_idx
        #     projected_dist[range(self.batch_size), lower_idx[eq_mask]] += next_dist[
        #         eq_mask
        #     ]

        #     ne_mask = ~eq_mask
        #     l_idx = lower_idx[ne_mask]
        #     u_idx = upper_idx[ne_mask]
        #     b_val = b[ne_mask]
        #     proj_batch = torch.arange(self.batch_size).to(self.device)[ne_mask]

        #     projected_dist[proj_batch, l_idx] += next_dist[ne_mask] * (
        #         u_idx.float() - b_val
        #     )
        #     projected_dist[proj_batch, u_idx] += next_dist[ne_mask] * (
        #         b_val - l_idx.float()
        #     )
        with torch.no_grad():
            # next_dist = self.target_net(next_states)  # [B, A, N]
            # next_q = torch.sum(next_dist * self.support, dim=2)
            # next_actions = next_q.argmax(dim=1)
            next_q_main = self.main_net.get_q_values(next_states)
            next_actions = next_q_main.argmax(dim=1)
            next_dist = self.target_net(next_states)
            next_dist = next_dist[range(self.batch_size), next_actions]  # [B, N]

            target_z = rewards.unsqueeze(1) + self.gamma * (
                1 - dones.unsqueeze(1)
            ) * self.support.unsqueeze(0)
            target_z = target_z.clamp(min=self.v_min, max=self.v_max)

            b = (target_z - self.v_min) / self.delta_z
            lower_idx = b.floor().long()
            upper_idx = b.ceil().long()

            # 初始化投影分布
            projected_dist = torch.zeros_like(next_dist)  # [B, N]

            batch_idx = (
                torch.arange(self.batch_size)
                .unsqueeze(1)
                .expand(-1, self.atom_size)
                .to(self.device)
            )

            # 等下标情况：lower == upper
            eq_mask = lower_idx == upper_idx
            projected_dist[batch_idx[eq_mask], lower_idx[eq_mask]] += next_dist[eq_mask]

            # 不等下标情况
            ne_mask = lower_idx != upper_idx
            l_idx_ne = lower_idx[ne_mask]
            u_idx_ne = upper_idx[ne_mask]
            b_val_ne = b[ne_mask]
            weight = next_dist[ne_mask]
            batch_idx_ne = batch_idx[ne_mask]

            # projected_dist[batch_idx_ne, l_idx_ne] += weight * (
            #     u_idx_ne.float() - b_val_ne
            # )
            # projected_dist[batch_idx_ne, u_idx_ne] += weight * (
            #     b_val_ne - l_idx_ne.float()
            # )
            projected_dist[batch_idx_ne, l_idx_ne] += weight * (
                u_idx_ne.float() - b_val_ne
            )
            projected_dist[batch_idx_ne, u_idx_ne] += weight * (
                b_val_ne - l_idx_ne.float()
            )

        dist = self.main_net(states)
        log_p = torch.log(dist[range(self.batch_size), actions])
        # loss = -(log_p * projected_dist).sum(dim=1)
        # loss = (loss * weights).mean()
        per_sample_loss = -(log_p * projected_dist).sum(dim=1)
        loss = (per_sample_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 10.0)
        self.optimizer.step()

        with torch.no_grad():
            td_error = per_sample_loss.detach().cpu().numpy()
            self.memory.update_priorities(indices, td_error + 1e-6)

        self.main_net.reset_noise()
        self.target_net.reset_noise()

        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def push(self, *args):
        self.memory.push(*args)
