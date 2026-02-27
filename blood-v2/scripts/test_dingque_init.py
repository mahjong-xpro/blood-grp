#!/usr/bin/env python3
"""Test if model initialization has systematic bias towards action 33 (sou)."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add blood-v2/python to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

def test_initial_dingque_bias():
    """Test if freshly initialized model has bias towards action 33."""
    
    # Import after path setup
    from blood.model.factory import BloodActorCritic
    from sample_factory.algo.utils.context import global_model_factory
    from gymnasium import spaces
    from blood.env.blood_env import NUM_STUDENT_CHANNELS, ACTION_SPACE
    
    # Create minimal config object
    class MinimalConfig:
        def __init__(self):
            self.blood_obs_channels = NUM_STUDENT_CHANNELS
            self.blood_conv_ch = 256
            self.blood_num_blocks = 20
            self.blood_num_tile_attn_layers = 4
            self.blood_tile_attn_heads = 4
            self.use_rnn = False
            self.rnn_size = 512
            self.rnn_type = "lstm"
            self.rnn_num_layers = 2
            self.aux_shanten_weight = 1.0
            self.aux_opp_waits_weight = 0.3
            self.oracle_enabled = False
            self.opponent_predictor_enabled = False
            self.turn_attention_enabled = False
    
    cfg = MinimalConfig()
    
    # Create observation and action spaces
    obs_space = spaces.Dict({
        "obs": spaces.Box(0, 1, (NUM_STUDENT_CHANNELS * 27,), dtype=np.float32),
        "action_mask": spaces.Box(0, 1, (ACTION_SPACE,), dtype=np.float32),
    })
    action_space = spaces.Discrete(ACTION_SPACE)
    
    # Create model
    model_factory = global_model_factory()
    model = BloodActorCritic(model_factory, obs_space, action_space, cfg)
    model.eval()
    
    # Create fake dingque observation (only actions 31-33 are legal)
    batch_size = 1000
    obs = torch.zeros(batch_size, NUM_STUDENT_CHANNELS * 27)
    mask = torch.zeros(batch_size, ACTION_SPACE)
    mask[:, 31:34] = 1.0  # Only dingque actions are legal
    
    obs_dict = {"obs": obs, "action_mask": mask}
    rnn_states = torch.zeros(batch_size, 2, model.core.get_out_size()) if cfg.use_rnn else None
    
    # Forward pass
    with torch.no_grad():
        result = model(obs_dict, rnn_states, values_only=False)
        logits = result["action_logits"]
        
        # Get dingque logits (31=man, 32=pin, 33=sou)
        dq_logits = logits[:, 31:34]
        probs = torch.softmax(dq_logits, dim=-1)
        
        # Sample actions
        actions = torch.multinomial(probs, 1).squeeze(-1)
        
        # Count distribution
        man_count = (actions == 0).sum().item()
        pin_count = (actions == 1).sum().item()
        sou_count = (actions == 2).sum().item()
        
        print("=" * 60)
        print("DINGQUE INITIALIZATION BIAS TEST")
        print("=" * 60)
        print(f"\nSampled {batch_size} dingque decisions from freshly initialized model:")
        print(f"  Man (action 31): {man_count:4d} ({man_count/batch_size*100:5.1f}%)")
        print(f"  Pin (action 32): {pin_count:4d} ({pin_count/batch_size*100:5.1f}%)")
        print(f"  Sou (action 33): {sou_count:4d} ({sou_count/batch_size*100:5.1f}%)")
        print(f"\nMean logits (before softmax):")
        print(f"  Man: {dq_logits[:, 0].mean().item():+.6f}")
        print(f"  Pin: {dq_logits[:, 1].mean().item():+.6f}")
        print(f"  Sou: {dq_logits[:, 2].mean().item():+.6f}")
        print(f"\nMean probabilities (after softmax):")
        print(f"  Man: {probs[:, 0].mean().item():.6f}")
        print(f"  Pin: {probs[:, 1].mean().item():.6f}")
        print(f"  Sou: {probs[:, 2].mean().item():.6f}")
        
        # Check if there's significant bias
        expected = batch_size / 3
        chi_sq = sum((count - expected)**2 / expected 
                     for count in [man_count, pin_count, sou_count])
        
        print(f"\nChi-square test (expected {expected:.1f} each):")
        print(f"  χ² = {chi_sq:.2f}")
        if chi_sq > 5.99:  # p < 0.05, df=2
            print("  ⚠️  SIGNIFICANT BIAS DETECTED (p < 0.05)")
            if sou_count > man_count and sou_count > pin_count:
                print("  🔴 BIAS TOWARDS SOU (action 33) - THIS EXPLAINS THE BUG!")
        else:
            print("  ✅ No significant bias")
        
        print("=" * 60)

if __name__ == "__main__":
    test_initial_dingque_bias()