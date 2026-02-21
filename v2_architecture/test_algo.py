import torch
from algo_core.network import PPOActorCritic
from algo_core.ppo import PPOAgent
from algo_core.oracle import OracleTrainer

def test_pipeline():
    # 1. Create dummy inputs (Batch=2, Channels=423, Width=27)
    obs_student = torch.randn(2, 423, 27)
    obs_teacher = torch.randn(2, 544, 27) # Teacher sees more (e.g. 544 channels)
    mask = torch.ones(2, 35, dtype=torch.bool)
    
    # 2. Initialize networks
    student_net = PPOActorCritic(obs_channels=423)
    teacher_net = PPOActorCritic(obs_channels=544)
    
    # 3. Initialize Agents
    student_agent = PPOAgent(student_net)
    teacher_agent = PPOAgent(teacher_net) # Teacher would be pre-trained and frozen
    
    # 4. Dummy Rollout Data
    actions = torch.tensor([5, 12])
    returns = torch.tensor([1.5, -0.5])
    advantages = torch.tensor([0.8, -0.2])
    log_probs_old = torch.randn(2)
    
    # 5. Distillation Training Step
    trainer = OracleTrainer(student_agent, teacher_agent, distill_weight=0.5)
    loss_stats = trainer.train_step(obs_student, obs_teacher, actions, returns, advantages, log_probs_old)
    
    print("Optimization Step Successful!")
    print("Loss Stats:", loss_stats)

if __name__ == "__main__":
    test_pipeline()
