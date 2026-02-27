"""Main training entry point for Sample Factory v2."""

import sys
import os
import signal
import atexit
import logging
from pathlib import Path

import torch
import yaml
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner

# Import learner_patch FIRST to apply patches at module import time.
# This ensures patches are active in both main process and spawned workers.
from . import learner_patch  # noqa: F401

from ..cfg import add_blood_args, blood_override_defaults
from ..env.blood_env import BloodMahjongEnv
from ..model.factory import register_blood_model
from .callbacks import BloodObserver

log = logging.getLogger(__name__)


def _setup_process_cleanup():
    """Kill all SF2 worker processes when the main process exits.

    SF2 spawns worker processes via multiprocessing.spawn. These are NOT
    daemon processes and won't die automatically when the parent crashes
    (e.g. CUDA OOM). This registers an atexit + SIGTERM handler that sends
    SIGTERM to the entire process group, cleaning up orphaned workers.

    os.setpgrp() makes this process the group leader so that all SF2
    workers (spawned after this call) inherit the same process group.
    """
    try:
        os.setpgrp()
    except OSError:
        pass  # Already a group leader or not permitted

    _pgid = os.getpgid(0)
    _main_pid = os.getpid()

    def _cleanup(signum=None, frame=None):
        if os.getpid() != _main_pid:
            return
        try:
            os.killpg(_pgid, signal.SIGTERM)
        except Exception:
            pass

    atexit.register(_cleanup)
    try:
        signal.signal(signal.SIGTERM, _cleanup)
    except (OSError, ValueError):
        pass  # Can't set signal in non-main thread


def _build_argv_from_yaml(config_path: str, original_argv: list[str]) -> list[str]:
    """Build a new argv list by merging YAML config values with CLI args.

    Does NOT mutate sys.argv. Returns a new list.

    Args:
        config_path: Path to the YAML config file
        original_argv: The original sys.argv to merge with

    Returns:
        New argv list with YAML values injected
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        return list(original_argv)

    # Start with a copy of original argv
    new_argv = list(original_argv)

    # Keys handled outside SF2 arg parsing — skip them here.
    # encoder_custom: not a registered SF2 arg.
    # oracle_enabled / league_enabled: have dedicated --no_xxx counterparts;
    #   handled below via explicit --no_oracle / --no_league injection.
    _skip = {"encoder_custom", "oracle_enabled", "league_enabled"}

    # SF2 args that use type=str2bool (accept --key False to disable).
    # store_true args cannot be set to False via CLI, so we skip them when False.
    _str2bool_args = {"use_rnn", "normalize_input", "normalize_returns"}

    # Inject --no_oracle / --no_league if yaml explicitly disables them.
    # (Default is True for both; only inject the negation flag when False.)
    if cfg.get("oracle_enabled") is False and "--no_oracle" not in new_argv:
        new_argv.append("--no_oracle")
    if cfg.get("league_enabled") is False and "--no_league" not in new_argv:
        new_argv.append("--no_league")

    # Append yaml values as CLI args (only if not already in new_argv)
    existing = set(new_argv)
    for key, val in cfg.items():
        if key in _skip:
            continue
        flag = f"--{key}"
        if flag in existing:
            continue
        if isinstance(val, bool):
            if val:
                if key in _str2bool_args:
                    # str2bool args require an explicit value (--use_rnn True)
                    new_argv.extend([flag, "True"])
                else:
                    new_argv.append(flag)  # store_true: no value needed
            elif key in _str2bool_args:
                # str2bool args: pass --key False to override set_defaults(key=True)
                new_argv.extend([flag, "False"])
            # else: store_true arg with False value — skip (can't unset via CLI)
        elif isinstance(val, list):
            # nargs="*" args (e.g. normalize_input_keys): pass each element separately
            # --normalize_input_keys obs oracle_obs
            new_argv.extend([flag] + [str(v) for v in val])
        else:
            new_argv.extend([flag, str(val)])

    return new_argv


def _extract_config_path(argv: list[str]) -> tuple[str | None, list[str]]:
    """Extract --config <path> from argv, returning (path, cleaned_argv).

    Returns (None, original_argv) if --config is not present.
    """
    args = argv[1:]  # skip argv[0] (program name)
    if "--config" not in args:
        return None, list(argv)
    idx = args.index("--config")
    if idx + 1 >= len(args):
        return None, list(argv)
    yaml_path = args[idx + 1]
    # Remove --config <path> from argv
    cleaned = [argv[0]] + args[:idx] + args[idx + 2:]
    return yaml_path, cleaned


def make_blood_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    opp_mode = getattr(cfg, "opponent_mode", "rulebot")
    if opp_mode == "selfplay":
        from blood.env.selfplay_env import SelfPlayEnv
        return SelfPlayEnv(cfg=cfg)
    return BloodMahjongEnv(cfg=cfg)


def register_blood_components():
    register_env("blood_mahjong", make_blood_env)
    register_blood_model()


def _configure_logging():
    """Reduce SF2 log noise: suppress model architecture dump and worker chatter."""
    import logging
    # SF2 prints full model architecture at INFO — suppress to WARNING
    logging.getLogger("sample_factory").setLevel(logging.WARNING)
    # Re-enable key SF2 loggers we do want to see
    for name in (
        "sample_factory.runner",
        "sample_factory.algo.learning.learner",
        "sample_factory.algo.runners",
    ):
        logging.getLogger(name).setLevel(logging.INFO)
    # Always show blood logs
    logging.getLogger("blood").setLevel(logging.INFO)


def _convert_numpy_scalars(obj, _memo=None):
    """Recursively convert numpy scalars to Python native types in-place.

    PyTorch 2.6+ defaults to weights_only=True in torch.load, which rejects
    numpy scalars (numpy.core.multiarray.scalar).  This converts them to
    int/float so the checkpoint can be loaded safely.
    """
    import numpy as np
    if _memo is None:
        _memo = set()
    obj_id = id(obj)
    if obj_id in _memo:
        return obj
    _memo.add(obj_id)

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _convert_numpy_scalars(v, _memo)
    elif isinstance(obj, (list, tuple)):
        converted = [_convert_numpy_scalars(item, _memo) for item in obj]
        if isinstance(obj, tuple):
            return tuple(converted)
        obj[:] = converted
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    return obj


# _patch_learner_load_state moved to learner_patch.py (applied at import time)


def _seed_init_checkpoint(cfg):
    """Seed a previous phase's checkpoint into the new experiment directory.

    SF2's learner runs in a subprocess, so we can't load weights directly from
    the main process.  Instead, we copy the full checkpoint (model + optimizer)
    into the target experiment's checkpoint_p0/ directory *before*
    runner.init().  SF2 will then discover and load it via its built-in
    checkpoint restoration logic.

    We preserve optimizer state (SF2's _load_state requires it) and reset
    env_steps/train_steps to 0 so the new phase starts from step 0.
    The new phase's LR schedule will override the optimizer's LR.

    If the target directory already contains checkpoints (e.g. from a previous
    run), we skip to avoid overwriting real training progress.
    """
    import glob
    from sample_factory.utils.utils import experiment_dir

    init_path = getattr(cfg, "init_checkpoint_path", "")
    if not init_path:
        return

    init_path = Path(init_path)
    if not init_path.exists():
        log.warning("init_checkpoint_path does not exist: %s — skipping", init_path)
        return

    # Determine target directory
    exp_dir = Path(experiment_dir(cfg=cfg))
    ckpt_dir = exp_dir / "checkpoint_p0"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # If target already has checkpoints, don't overwrite — user should use --resume
    existing = glob.glob(str(ckpt_dir / "checkpoint_*.pth"))
    if existing:
        log.info("Target checkpoint dir already has %d checkpoint(s); "
                 "skipping init_checkpoint_path copy (use --resume to continue)",
                 len(existing))
        return

    log.info("Loading init checkpoint from previous phase: %s", init_path)
    source_ckpt = torch.load(str(init_path), map_location="cpu", weights_only=False)

    # Build a new checkpoint preserving model + optimizer state.
    # SF2 checkpoint keys: model, optimizer, env_steps, stats, cfg, ...
    # We keep the optimizer state because SF2's _load_state unconditionally
    # loads it (KeyError if missing).
    seed_ckpt = dict(source_ckpt)  # shallow copy
    # Mark as a cross-phase seed so _patched_load_state can distinguish it
    # from a normal resume checkpoint (Issue #R9-H4).
    seed_ckpt["_cross_phase_seed"] = True
    # Reset training counters so the new phase starts from step 0
    seed_ckpt["env_steps"] = 0
    seed_ckpt["train_step"] = 0

    # Reset optimizer LR to the new phase's configured learning_rate.
    # Without this, the inherited LR (potentially inflated to 0.01 by
    # kl_adaptive_minibatch) causes value_loss spikes at phase transitions.
    new_lr = getattr(cfg, "learning_rate", None)
    opt_state = seed_ckpt.get("optimizer")
    if new_lr is not None and opt_state is not None:
        param_groups = opt_state.get("param_groups", [])
        for pg in param_groups:
            old_lr = pg.get("lr", "?")
            pg["lr"] = new_lr
            pg["initial_lr"] = new_lr
        if param_groups:
            log.info("Reset optimizer LR: %s → %s (new phase learning_rate)", old_lr, new_lr)

    # Convert numpy scalars to Python native types so that torch.load with
    # weights_only=True (PyTorch 2.6+ default) can deserialize the checkpoint.
    # SF2 stores numpy scalars in stats/cfg which fail safe unpickling.
    _convert_numpy_scalars(seed_ckpt)

    dest = ckpt_dir / "checkpoint_000000000_0.pth"
    log.info("Seeding init checkpoint: %s → %s", init_path, dest)
    torch.save(seed_ckpt, str(dest))
    log.info("Init checkpoint seeded successfully (model + optimizer, counters reset to 0)")


def run_training():
    _setup_process_cleanup()
    register_blood_components()
    _configure_logging()

    # Expand --config <yaml> into individual SF2 CLI args before SF2 parses sys.argv.
    # Build a merged argv without mutating the global sys.argv permanently.
    config_path, cleaned_argv = _extract_config_path(sys.argv)
    if config_path is not None:
        merged_argv = _build_argv_from_yaml(config_path, cleaned_argv)
    else:
        merged_argv = cleaned_argv

    # SF2 requires --env as a mandatory CLI argument before set_defaults can run.
    if "--env" not in merged_argv:
        merged_argv.extend(["--env", "blood_mahjong"])

    # SF2's argument parser reads from sys.argv, so temporarily swap and restore.
    original_argv = sys.argv
    try:
        sys.argv = merged_argv
        parser, partial_cfg = parse_sf_args(evaluation=False)
        add_blood_args(parser)
        blood_override_defaults(parser)
        cfg = parse_full_cfg(parser)
    finally:
        sys.argv = original_argv  # Always restore, even on parse errors

    # PERFORMANCE OPTIMIZATION: Enable parallel mode for 10-15x speedup.
    #
    # Previously forced serial_mode=True because monkey-patches applied in main
    # process weren't inherited by SF2's spawned workers (multiprocessing.spawn).
    #
    # NEW APPROACH: learner_patch.py applies patches at module import time,
    # ensuring they're active in ALL processes (main + workers). This allows
    # us to use SF2's parallel mode while keeping all custom losses.
    #
    # Expected improvement: 18.6 steps/sec → 200-300 steps/sec (10-15x faster)
    # Training time: 10.6 days → 1-2 days for full curriculum
    #
    # If you see "BloodLossComputer initialized" logs from multiple PIDs,
    # the patches are working correctly in parallel mode.
    log.info("Parallel mode enabled - expecting 10-15x speedup vs serial mode")
    log.info("Monitor for '[LearnerPatch]' logs from worker processes to verify patches are active")

    cfg, runner = make_runner(cfg)

    observer = BloodObserver(cfg)
    runner.register_observer(observer)

    # Cross-phase checkpoint chaining: seed previous phase's weights before SF2 init
    _seed_init_checkpoint(cfg)

    runner.init()

    status = runner.run()
    return status


if __name__ == "__main__":
    sys.exit(run_training())
