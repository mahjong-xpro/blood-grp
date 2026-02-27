#!/usr/bin/env python3
"""
Verify that all performance optimizations are properly configured.

Usage:
    python scripts/verify_optimization.py [--config configs/warmup.yaml]
"""

import argparse
import sys
import yaml
from pathlib import Path

def check_config(config_path):
    """Check if config has performance optimizations enabled."""
    print(f"\n{'='*60}")
    print(f"Checking: {config_path}")
    print(f"{'='*60}\n")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    issues = []
    warnings = []
    
    # Check mixed precision
    if cfg.get('use_mixed_precision', False):
        print("✅ Mixed precision training: ENABLED")
    else:
        issues.append("❌ Mixed precision training: DISABLED")
        print(issues[-1])
    
    # Check batch size
    batch_size = cfg.get('batch_size', 0)
    if batch_size >= 2048:
        print(f"✅ Batch size: {batch_size} (optimized)")
    else:
        warnings.append(f"⚠️  Batch size: {batch_size} (consider increasing to 2048+)")
        print(warnings[-1])
    
    # Check save frequency
    save_every = cfg.get('save_every_sec', 0)
    if save_every >= 300:
        print(f"✅ Save frequency: {save_every}s (reduced I/O overhead)")
    else:
        warnings.append(f"⚠️  Save frequency: {save_every}s (consider 300+ for less I/O)")
        print(warnings[-1])
    
    # Check eval frequency
    eval_every = cfg.get('blood_arena_eval_every', 0)
    if eval_every >= 500000:
        print(f"✅ Eval frequency: {eval_every} steps (reduced overhead)")
    elif eval_every > 0:
        warnings.append(f"⚠️  Eval frequency: {eval_every} steps (consider 500k+ for less overhead)")
        print(warnings[-1])
    
    # Check num_workers and num_envs
    num_workers = cfg.get('num_workers', 0)
    num_envs = cfg.get('num_envs_per_worker', 0)
    total_envs = num_workers * num_envs
    print(f"\n📊 Parallelism:")
    print(f"   Workers: {num_workers}")
    print(f"   Envs per worker: {num_envs}")
    print(f"   Total parallel envs: {total_envs}")
    
    if total_envs >= 256:
        print(f"   ✅ Good parallelism (256+ envs)")
    else:
        warnings.append(f"   ⚠️  Low parallelism ({total_envs} envs, consider 256+)")
        print(warnings[-1])
    
    return issues, warnings


def check_learner_patch():
    """Check if learner_patch.py is properly configured."""
    print(f"\n{'='*60}")
    print("Checking: learner_patch.py")
    print(f"{'='*60}\n")
    
    patch_file = Path("python/blood/training/learner_patch.py")
    if not patch_file.exists():
        print("❌ learner_patch.py not found!")
        return ["learner_patch.py missing"]
    
    content = patch_file.read_text()
    
    issues = []
    
    # Check for new PyTorch API
    if "torch.amp.GradScaler('cuda')" in content:
        print("✅ Using new PyTorch AMP API (no warnings)")
    elif "torch.cuda.amp.GradScaler()" in content:
        issues.append("⚠️  Using deprecated PyTorch AMP API (will show warnings)")
        print(issues[-1])
    
    # Check for module-level patching
    if "Learner._calculate_losses = _patched_calculate_losses" in content:
        print("✅ Module-level patching: ENABLED (parallel mode compatible)")
    else:
        issues.append("❌ Module-level patching: NOT FOUND")
        print(issues[-1])
    
    return issues


def check_runner():
    """Check if runner.py has serial_mode restriction removed."""
    print(f"\n{'='*60}")
    print("Checking: runner.py")
    print(f"{'='*60}\n")
    
    runner_file = Path("python/blood/training/runner.py")
    if not runner_file.exists():
        print("❌ runner.py not found!")
        return ["runner.py missing"]
    
    content = runner_file.read_text()
    
    issues = []
    
    # Check for serial_mode restriction
    if "cfg.serial_mode = True" in content:
        issues.append("❌ Serial mode restriction STILL PRESENT (parallel mode disabled!)")
        print(issues[-1])
    else:
        print("✅ Serial mode restriction: REMOVED (parallel mode enabled)")
    
    # Check for learner_patch import
    if "import blood.training.learner_patch" in content or "from blood.training import learner_patch" in content:
        print("✅ learner_patch import: FOUND")
    else:
        issues.append("⚠️  learner_patch import: NOT FOUND (patches may not apply)")
        print(issues[-1])
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Verify Blood-v2 performance optimizations")
    parser.add_argument("--config", default="configs/warmup.yaml", help="Config file to check")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Blood-v2 Performance Optimization Verification")
    print("="*60)
    
    all_issues = []
    all_warnings = []
    
    # Check config
    issues, warnings = check_config(args.config)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    # Check learner_patch
    issues = check_learner_patch()
    all_issues.extend(issues)
    
    # Check runner
    issues = check_runner()
    all_issues.extend(issues)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    
    if not all_issues and not all_warnings:
        print("✅ All optimizations properly configured!")
        print("\nExpected performance:")
        print("  - Training speed: 280-560 steps/sec (15-30x speedup)")
        print("  - Warmup phase: 1-2 hours (vs 30 hours baseline)")
        print("  - Full curriculum: 10-17 hours (vs 10.6 days baseline)")
        return 0
    
    if all_warnings:
        print("⚠️  Warnings:")
        for w in all_warnings:
            print(f"  {w}")
        print()
    
    if all_issues:
        print("❌ Issues found:")
        for i in all_issues:
            print(f"  {i}")
        print("\nPlease fix these issues before training.")
        return 1
    
    print("✅ No critical issues, but consider addressing warnings for optimal performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())