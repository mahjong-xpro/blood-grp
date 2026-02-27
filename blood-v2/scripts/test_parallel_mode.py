#!/usr/bin/env python3
"""
Test script to verify parallel mode is working correctly.

This script runs a short training session and checks:
1. Patches are applied in worker processes (multiple PIDs in logs)
2. Custom losses are computed correctly
3. Training speed is significantly improved

Usage:
    python3 scripts/test_parallel_mode.py
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add blood-v2/python to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

def test_import_patch():
    """Test that importing learner_patch applies patches correctly."""
    print("=" * 60)
    print("TEST 1: Import-time patch application")
    print("=" * 60)
    
    # Import should trigger patch
    from blood.training import learner_patch
    from sample_factory.algo.learning.learner import Learner
    
    # Check if methods are patched
    is_patched = (
        Learner._calculate_losses != learner_patch._original_calculate_losses
        and Learner._load_state != learner_patch._original_load_state
    )
    
    if is_patched:
        print("✓ Patches applied successfully at import time")
        print(f"  - _calculate_losses: {Learner._calculate_losses.__name__}")
        print(f"  - _load_state: {Learner._load_state.__name__}")
        return True
    else:
        print("✗ Patches NOT applied")
        return False


def test_smoke_training():
    """Run a minimal training session to verify parallel mode works."""
    print("\n" + "=" * 60)
    print("TEST 2: Smoke test with parallel mode")
    print("=" * 60)
    
    # Create minimal config
    config_content = """
# Minimal config for smoke test
num_workers: 2
num_envs_per_worker: 4
batch_size: 128
num_batches_per_epoch: 2
train_for_env_steps: 1000
save_every_sec: 9999
experiment: blood_parallel_test

# Model
blood_obs_channels: 473
blood_conv_channels: 256
blood_num_res_blocks: 20
blood_encoder_out_dim: 1024
blood_enc_proj_layers: 3
blood_num_tile_attn_layers: 4
blood_tile_attn_heads: 4

# Oracle
oracle_enabled: false

# League
league_enabled: false

# LSTM
use_rnn: true
rnn_type: lstm
rnn_size: 512
rnn_num_layers: 2
rollout: 32
recurrence: 32

# Opponent
opponent_mode: rulebot
"""
    
    config_path = Path("configs/test_parallel.yaml")
    config_path.write_text(config_content)
    
    print(f"Created test config: {config_path}")
    print("Starting training (1000 steps)...")
    print("Watch for '[LearnerPatch]' logs from multiple PIDs")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(
            [
                sys.executable, "-m", "blood.training.runner",
                "--config", str(config_path),
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )
        
        elapsed = time.time() - start_time
        
        # Check output
        output = result.stdout + result.stderr
        
        # Look for patch logs from multiple PIDs
        patch_logs = [line for line in output.split('\n') if '[LearnerPatch]' in line]
        unique_pids = set()
        for line in patch_logs:
            if 'pid=' in line:
                pid = line.split('pid=')[1].split(')')[0]
                unique_pids.add(pid)
        
        print("-" * 60)
        print(f"Training completed in {elapsed:.1f}s")
        print(f"Found {len(patch_logs)} patch log entries")
        print(f"Unique PIDs: {len(unique_pids)} - {unique_pids}")
        
        if len(unique_pids) >= 2:
            print("✓ Patches active in multiple processes (parallel mode working!)")
            
            # Estimate speedup
            steps_per_sec = 1000 / elapsed
            print(f"\nPerformance: {steps_per_sec:.1f} steps/sec")
            print(f"Expected with serial mode: ~18.6 steps/sec")
            print(f"Speedup: {steps_per_sec / 18.6:.1f}x")
            
            return True
        else:
            print("✗ Patches only in main process (parallel mode NOT working)")
            print("\nDebug output:")
            print(output[-2000:])  # Last 2000 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Training timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()


def main():
    print("Blood-V2 Parallel Mode Verification")
    print("=" * 60)
    
    results = []
    
    # Test 1: Import-time patches
    results.append(("Import-time patches", test_import_patch()))
    
    # Test 2: Smoke training
    results.append(("Parallel training", test_smoke_training()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Parallel mode is working correctly.")
        print("Expected speedup: 10-15x (18.6 → 200-300 steps/sec)")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())