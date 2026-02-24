"""Hotfix script for corrupted competitive checkpoint.

Resets the components that were damaged by the oracle_value_distill bug:
  1. Student critic weights  → restored from warmup checkpoint
  2. Oracle value head       → re-initialized (was random anyway)
  3. Optimizer state         → cleared (resets LR from 0.01 back to config value)

Keeps everything else (student encoder, LSTM, oracle policy head, league pool).

Usage:
    python scripts/hotfix_checkpoint.py \
        --competitive  path/to/train_dir/blood_v2_competitive/checkpoint_p0/checkpoint_p0_XXXXXXX.pth \
        --warmup       path/to/train_dir/blood_v2_warmup/checkpoint_p0/checkpoint_p0_XXXXXXX.pth \
        --output       path/to/train_dir/blood_v2_competitive/checkpoint_p0/checkpoint_hotfixed.pth

Then restart competitive training pointing at the hotfixed checkpoint.
"""

import argparse
import copy
import torch


def hotfix(competitive_path: str, warmup_path: str, output_path: str) -> None:
    print(f"Loading competitive checkpoint: {competitive_path}")
    comp_ckpt = torch.load(competitive_path, map_location="cpu", weights_only=False)

    print(f"Loading warmup checkpoint:      {warmup_path}")
    warm_ckpt = torch.load(warmup_path, map_location="cpu", weights_only=False)

    # SF2 stores model weights under the "model" key
    comp_state = comp_ckpt.get("model", comp_ckpt)
    warm_state = warm_ckpt.get("model", warm_ckpt)

    fixed_state = copy.deepcopy(comp_state)
    reset_keys = []
    reinit_keys = []

    # 1. Reset student critic weights from warmup checkpoint.
    #    The critic was trained to match random oracle values for 1.2M steps.
    #    Warmup critic is clean (no oracle corruption).
    for key in list(comp_state.keys()):
        if "critic_head" in key or "critic_linear" in key:
            if key in warm_state:
                fixed_state[key] = warm_state[key].clone()
                reset_keys.append(key)
            else:
                print(f"  WARNING: critic key not found in warmup: {key}")

    # 2. Re-initialize oracle value head.
    #    It was never trained (zero gradient path), so it's random init anyway.
    #    Xavier uniform for weight tensors, zeros for bias.
    for key in list(comp_state.keys()):
        if "oracle_encoder.value_head" in key:
            t = fixed_state[key].clone()
            if t.dim() >= 2:
                torch.nn.init.xavier_uniform_(t)
            else:
                torch.nn.init.zeros_(t)
            fixed_state[key] = t
            reinit_keys.append(key)

    print(f"\nReset from warmup ({len(reset_keys)} keys):")
    for k in reset_keys:
        print(f"  {k}")

    print(f"\nRe-initialized oracle value head ({len(reinit_keys)} keys):")
    for k in reinit_keys:
        print(f"  {k}")

    # 3. Clear optimizer state so LR resets to the value in competitive.yaml.
    #    The Adam m/v accumulators encode the history of bad gradients.
    fixed_ckpt = copy.deepcopy(comp_ckpt)
    fixed_ckpt["model"] = fixed_state

    if "optimizer" in fixed_ckpt:
        opt = fixed_ckpt["optimizer"]
        if "state" in opt:
            n_cleared = len(opt["state"])
            opt["state"] = {}
            print(f"\nCleared optimizer state ({n_cleared} parameter groups)")
        # Reset step counter so Adam starts fresh
        if "param_groups" in opt:
            for pg in opt["param_groups"]:
                pg["step"] = 0 if "step" in pg else pg.get("step", 0)
    else:
        print("\nWARNING: no 'optimizer' key found in checkpoint")

    # 4. Keep env_steps so league pool and league_add_every logic continues correctly.
    #    Do NOT reset env_steps — the league pool is already partially filled.
    env_steps = fixed_ckpt.get("env_steps", "unknown")
    print(f"\nKeeping env_steps={env_steps} (league pool preserved)")

    torch.save(fixed_ckpt, output_path)
    print(f"\nSaved hotfixed checkpoint → {output_path}")
    print("\nNext steps:")
    print("  1. Verify competitive.yaml has all fixes applied (lr_schedule_kl_threshold, etc.)")
    print("  2. Restart competitive training with --load_checkpoint_kind=last")
    print("     pointing at the hotfixed checkpoint directory")


def main():
    parser = argparse.ArgumentParser(description="Hotfix corrupted competitive checkpoint")
    parser.add_argument("--competitive", required=True,
                        help="Path to corrupted competitive checkpoint .pth")
    parser.add_argument("--warmup", required=True,
                        help="Path to clean warmup checkpoint .pth")
    parser.add_argument("--output", required=True,
                        help="Output path for hotfixed checkpoint .pth")
    args = parser.parse_args()

    hotfix(args.competitive, args.warmup, args.output)


if __name__ == "__main__":
    main()
