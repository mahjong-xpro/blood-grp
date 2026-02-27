#!/usr/bin/env python3
"""Test if action logits layer has systematic bias towards action 33 (sou)."""

import torch
import torch.nn as nn

def test_action_layer_init():
    """Test if a freshly initialized Linear layer has bias towards last action."""
    
    print("=" * 60)
    print("ACTION LOGITS LAYER INITIALIZATION TEST")
    print("=" * 60)
    
    # Simulate the final action layer: Linear(512 -> 34)
    # This is what Sample Factory creates in action_parameterization
    action_layer = nn.Linear(512, 34)
    
    # Test with orthogonal initialization (what Blood-V2 uses)
    nn.init.orthogonal_(action_layer.weight, gain=0.01)
    nn.init.zeros_(action_layer.bias)
    
    # Create fake input (batch of 1000 samples)
    batch_size = 1000
    fake_input = torch.randn(batch_size, 512) * 0.1  # Small random features
    
    # Forward pass
    with torch.no_grad():
        logits = action_layer(fake_input)
        
        # Get dingque logits (31=man, 32=pin, 33=sou)
        dq_logits = logits[:, 31:34]
        probs = torch.softmax(dq_logits, dim=-1)
        
        # Sample actions
        actions = torch.multinomial(probs, 1).squeeze(-1)
        
        # Count distribution
        man_count = (actions == 0).sum().item()
        pin_count = (actions == 1).sum().item()
        sou_count = (actions == 2).sum().item()
        
        print(f"\nOrthogonal init (gain=0.01) + zero bias:")
        print(f"  Man (action 31): {man_count:4d} ({man_count/batch_size*100:5.1f}%)")
        print(f"  Pin (action 32): {pin_count:4d} ({pin_count/batch_size*100:5.1f}%)")
        print(f"  Sou (action 33): {sou_count:4d} ({sou_count/batch_size*100:5.1f}%)")
        
        print(f"\nMean logits:")
        print(f"  Man: {dq_logits[:, 0].mean().item():+.6f}")
        print(f"  Pin: {dq_logits[:, 1].mean().item():+.6f}")
        print(f"  Sou: {dq_logits[:, 2].mean().item():+.6f}")
        
        print(f"\nLogit std dev:")
        print(f"  Man: {dq_logits[:, 0].std().item():.6f}")
        print(f"  Pin: {dq_logits[:, 1].std().item():.6f}")
        print(f"  Sou: {dq_logits[:, 2].std().item():.6f}")
        
        # Check weight norms for actions 31-33
        weight_norms = torch.norm(action_layer.weight, dim=1)
        print(f"\nWeight L2 norms:")
        print(f"  Man (31): {weight_norms[31].item():.6f}")
        print(f"  Pin (32): {weight_norms[32].item():.6f}")
        print(f"  Sou (33): {weight_norms[33].item():.6f}")
        
        # Chi-square test
        expected = batch_size / 3
        chi_sq = sum((count - expected)**2 / expected 
                     for count in [man_count, pin_count, sou_count])
        
        print(f"\nChi-square test (expected {expected:.1f} each):")
        print(f"  χ² = {chi_sq:.2f}")
        if chi_sq > 5.99:  # p < 0.05, df=2
            print("  ⚠️  SIGNIFICANT BIAS DETECTED (p < 0.05)")
            if sou_count > man_count and sou_count > pin_count:
                print("  🔴 BIAS TOWARDS SOU (action 33)")
        else:
            print("  ✅ No significant bias")
    
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT INITIALIZATIONS")
    print("=" * 60)
    
    # Test with default PyTorch init
    action_layer2 = nn.Linear(512, 34)
    # PyTorch default: Kaiming uniform
    
    with torch.no_grad():
        logits2 = action_layer2(fake_input)
        dq_logits2 = logits2[:, 31:34]
        probs2 = torch.softmax(dq_logits2, dim=-1)
        actions2 = torch.multinomial(probs2, 1).squeeze(-1)
        
        man_count2 = (actions2 == 0).sum().item()
        pin_count2 = (actions2 == 1).sum().item()
        sou_count2 = (actions2 == 2).sum().item()
        
        print(f"\nPyTorch default (Kaiming uniform):")
        print(f"  Man: {man_count2:4d} ({man_count2/batch_size*100:5.1f}%)")
        print(f"  Pin: {pin_count2:4d} ({pin_count2/batch_size*100:5.1f}%)")
        print(f"  Sou: {sou_count2:4d} ({sou_count2/batch_size*100:5.1f}%)")
        
        chi_sq2 = sum((count - expected)**2 / expected 
                      for count in [man_count2, pin_count2, sou_count2])
        print(f"  χ² = {chi_sq2:.2f}")
        if chi_sq2 > 5.99:
            print("  ⚠️  BIAS DETECTED")
        else:
            print("  ✅ No bias")
    
    print("=" * 60)

if __name__ == "__main__":
    # Run test multiple times to check consistency
    print("\nRunning 3 independent tests...\n")
    for i in range(3):
        print(f"\n{'='*60}")
        print(f"TEST RUN #{i+1}")
        print('='*60)
        test_action_layer_init()
        print()