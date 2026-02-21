import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

class SuitAwareConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False)
        self.num_suits = 3
        self.suit_width = 9

    def forward(self, x):
        # x is (B, C, 27)
        B, C, W = x.shape
        x_suits = x.view(B, C, self.num_suits, self.suit_width).transpose(1, 2).contiguous()
        x_suits = x_suits.view(B * self.num_suits, C, self.suit_width)
        x_padded = F.pad(x_suits, (1, 1))
        out = self.conv(x_padded)
        out_C = out.shape[1]
        out = out.view(B, self.num_suits, out_C, self.suit_width).transpose(1, 2).contiguous()
        out = out.view(B, out_C, W)
        return out

class PPOActorCritic(nn.Module):
    def __init__(self, obs_channels=423, conv_channels=192, embed_dim=256, num_heads=8, num_blocks=4, action_dim=35):
        super().__init__()
        self.stem = nn.Sequential(
            SuitAwareConv1d(obs_channels, conv_channels),
            nn.BatchNorm1d(conv_channels),
            nn.Mish()
        )
        
        # Project spatial 27 to sequences of 27 tokens
        self.proj = nn.Linear(conv_channels, embed_dim)
        
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, embed_dim * 4) for _ in range(num_blocks)]
        )
        
        # Shared features
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim * 27, 512),
            nn.Mish(),
            nn.LayerNorm(512)
        )

        # Actor head (Policy)
        self.actor_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, action_dim)
        )
        
        # Critic head (Value)
        self.critic_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, mask=None):
        # (B, C, 27)
        x = self.stem(obs)
        
        # (B, C, 27) -> (B, 27, C) -> (B, 27, embed_dim)
        x = x.transpose(1, 2)
        x = self.proj(x)
        
        # Transformer spatial mixing
        x = self.transformer(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        shared_feat = self.shared_mlp(x)
        
        # Actor
        logits = self.actor_head(shared_feat)
        if mask is not None:
            logits = logits.masked_fill(~mask, -float('inf'))
            
        probs = F.softmax(logits, dim=-1)
        
        # Critic
        value = self.critic_head(shared_feat)
        
        return probs, value, logits
