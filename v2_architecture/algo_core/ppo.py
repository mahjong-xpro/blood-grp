import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, network: nn.Module, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, entropy_coef=0.01, vf_coef=0.5):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

    def select_action(self, obs, mask=None):
        with torch.no_grad():
            probs, value, _ = self.network(obs, mask)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def update(self, rollouts):
        obs, actions, log_probs_old, returns, advantages = rollouts

        # Forward pass
        probs, values, logits = self.network(obs)
        dist = Categorical(probs)
        log_probs_new = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Policy Loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        obj1 = ratio * advantages
        obj2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(obj1, obj2).mean()

        # Value Loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)

        # Total Loss
        loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
