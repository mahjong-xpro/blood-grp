"""Oracle encoder and distillation loss.

The Oracle sees perfect information (all hands + wall) and is trained
jointly with the student. KL divergence from oracle to student policy
provides the distillation signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from blood.model.encoder import SuitAwareConv1d, BottleneckBlock, NUM_TILES, _num_groups

DEFAULT_ORACLE_CHANNELS = 430  # 384 student + 46 oracle extra


class OracleEncoder(nn.Module):
    """Oracle encoder: same architecture as student but with perfect info."""

    def __init__(
        self,
        obs_channels: int = DEFAULT_ORACLE_CHANNELS,
        conv_ch: int = 256,
        num_blocks: int = 20,
        out_dim: int = 1024,
        action_dim: int = 34,
    ):
        super().__init__()

        ng = _num_groups(conv_ch)
        layers = [
            SuitAwareConv1d(obs_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(BottleneckBlock(conv_ch))

        self.conv_stack = nn.Sequential(*layers)
        
        # Output Trunk
        trunk_dim = conv_ch * NUM_TILES
        
        # Policy Head (Teacher has its own head)
        self.policy_head = nn.Sequential(
            nn.Linear(trunk_dim, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, action_dim),
        )

    def forward(self, oracle_obs):
        B = oracle_obs.shape[0]
        # Reshape to (B, C, 27)
        x = oracle_obs.view(B, -1, NUM_TILES)
        x = self.conv_stack(x)
        x = x.reshape(B, -1)
        logits = self.policy_head(x)
        return logits


class DistillationLoss(nn.Module):
    """KL divergence distillation from oracle to student."""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, oracle_logits, action_mask=None):
        T = self.temperature

        if action_mask is not None:
            large_neg = torch.finfo(student_logits.dtype).min
            student_logits = student_logits.masked_fill(~action_mask, large_neg)
            oracle_logits = oracle_logits.masked_fill(~action_mask, large_neg)

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        oracle_probs = F.softmax(oracle_logits / T, dim=-1)

        kl = F.kl_div(student_log_probs, oracle_probs, reduction="batchmean")
        return kl * (T * T)
