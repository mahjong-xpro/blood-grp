import torch
import numpy as np
from network import PPOActorCritic
from ppo import PPOAgent
from oracle import OracleTrainer

# Note: In a real run, this would be imported from the compiled env_core module
# from env_core import PyBatchedEnv

class DummyBatchedEnv:
    """A dummy environment for testing the RL loop until Rust compilation is sorted."""
    def __init__(self, num_envs=16, obs_dim=423, action_dim=35):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
    def reset(self, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        return torch.randn(len(indices), self.obs_dim, 27), torch.ones(len(indices), self.action_dim, dtype=torch.bool)
        
    def step(self, actions):
        # Returns: obs, rewards, dones, masks
        obs = torch.randn(self.num_envs, self.obs_dim, 27)
        rewards = torch.randn(self.num_envs)
        dones = torch.rand(self.num_envs) > 0.95
        masks = torch.ones(self.num_envs, self.action_dim, dtype=torch.bool)
        return obs, rewards, dones, masks


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation.
    rewards: (T, B)
    values: (T, B)
    dones: (T, B)
    next_value: (B,)
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    last_gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t].float()
            next_v = next_value
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_v = values[t + 1]
            
        delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
        
    returns = advantages + values
    return returns, advantages


def train_loop():
    print("Initializing V2 Architecture Training Loop...")
    
    # Configuration
    NUM_ENVS = 16
    ROLLOUT_STEPS = 128
    EPOCHS = 5
    OBS_DIM = 423
    ACTION_DIM = 35
    
    # Initialize environment
    # env = PyBatchedEnv(NUM_ENVS)
    env = DummyBatchedEnv(NUM_ENVS, OBS_DIM, ACTION_DIM)
    
    # Initialize networks and agents
    student_net = PPOActorCritic(obs_channels=OBS_DIM)
    teacher_net = PPOActorCritic(obs_channels=544) # Oracle sees more
    
    student_agent = PPOAgent(student_net)
    teacher_agent = PPOAgent(teacher_net)
    trainer = OracleTrainer(student_agent, teacher_agent, distill_weight=0.5)
    
    # Buffers
    b_obs = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, OBS_DIM, 27)
    b_obs_oracle = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, 544, 27)
    b_actions = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, dtype=torch.long)
    b_logprobs = torch.zeros(ROLLOUT_STEPS, NUM_ENVS)
    b_rewards = torch.zeros(ROLLOUT_STEPS, NUM_ENVS)
    b_dones = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, dtype=torch.bool)
    b_values = torch.zeros(ROLLOUT_STEPS, NUM_ENVS)
    b_masks = torch.zeros(ROLLOUT_STEPS, NUM_ENVS, ACTION_DIM, dtype=torch.bool)
    
    obs, masks = env.reset()
    
    # --- Trajectory Collection ---
    print(f"Collecting {ROLLOUT_STEPS} steps across {NUM_ENVS} parallel environments...")
    for step in range(ROLLOUT_STEPS):
        b_obs[step] = obs
        b_masks[step] = masks
        
        # In a real setup, we'd also get oracle observations from the env
        oracle_obs = torch.randn(NUM_ENVS, 544, 27)
        b_obs_oracle[step] = oracle_obs
        
        # Agent decides
        with torch.no_grad():
            probs, values, _ = student_net(obs, masks)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            
        b_actions[step] = actions
        b_logprobs[step] = logprobs
        b_values[step] = values.squeeze()
        
        # Environment steps
        obs, rewards, dones, masks = env.step(actions.numpy())
        b_rewards[step] = rewards
        b_dones[step] = dones
        
    # --- GAE Computation ---
    print("Computing GAE and Returns...")
    with torch.no_grad():
        _, next_value, _ = student_net(obs, masks)
        next_value = next_value.squeeze()
        
    returns, advantages = compute_gae(b_rewards, b_values, b_dones, next_value)
    
    # Flatten buffers for PPO training
    # Shape becomes (ROLLOUT_STEPS * NUM_ENVS, ...)
    f_obs = b_obs.flatten(0, 1)
    f_obs_oracle = b_obs_oracle.flatten(0, 1)
    f_actions = b_actions.flatten(0, 1)
    f_returns = returns.flatten(0, 1)
    f_advantages = advantages.flatten(0, 1)
    f_logprobs = b_logprobs.flatten(0, 1)
    
    # Normalize advantages
    f_advantages = (f_advantages - f_advantages.mean()) / (f_advantages.std() + 1e-8)
    
    # --- PPO / Oracle Optimization ---
    print(f"Starting Policy Optimization ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        # A real implementation would use mini-batches here
        loss_stats = trainer.train_step(
            f_obs, 
            f_obs_oracle, 
            f_actions, 
            f_returns, 
            f_advantages, 
            f_logprobs
        )
        print(f"  Epoch {epoch+1}/{EPOCHS} | Policy Loss: {loss_stats['policy_loss']:.4f} | Distill Loss: {loss_stats['distill_loss']:.4f}")

    print("Training iteration complete!")

if __name__ == "__main__":
    train_loop()
