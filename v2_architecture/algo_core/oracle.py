import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_kl_loss(student_logits, teacher_logits, temperature=1.0):
    """
    Computes the KL divergence loss between student and teacher policies.
    Both inputs are expected to be raw logits.
    """
    # Soften the logits with temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL(Teacher || Student)
    # Average over batch dimension
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    # Scale by T^2 as per distillation literature
    return kl_loss * (temperature ** 2)

class OracleTrainer:
    def __init__(self, student_agent, teacher_agent, distill_weight=1.0, temperature=2.0):
        self.student = student_agent
        self.teacher = teacher_agent # This is usually frozen during distillation
        self.distill_weight = distill_weight
        self.temperature = temperature
        
    def train_step(self, obs_student, obs_teacher, actions, returns, advantages, log_probs_old):
        # 1. Get teacher's perfect distribution (no gradients needed)
        with torch.no_grad():
            _, _, teacher_logits = self.teacher.network(obs_teacher)
            
        # 2. Get student's distribution & RL loss components
        probs, values, student_logits = self.student.network(obs_student)
        dist = torch.distributions.Categorical(probs)
        log_probs_new = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO Policy Loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        obj1 = ratio * advantages
        obj2 = torch.clamp(ratio, 1 - self.student.clip_ratio, 1 + self.student.clip_ratio) * advantages
        policy_loss = -torch.min(obj1, obj2).mean()

        # PPO Value Loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # 3. Compute Distillation Loss
        distill_loss = compute_kl_loss(student_logits, teacher_logits, self.temperature)
        
        # 4. Total Loss
        total_loss = (policy_loss 
                      + self.student.vf_coef * value_loss 
                      - self.student.entropy_coef * entropy 
                      + self.distill_weight * distill_loss)

        # Optimize Student
        self.student.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.student.network.parameters(), 0.5)
        self.student.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'distill_loss': distill_loss.item(),
            'total_loss': total_loss.item()
        }
