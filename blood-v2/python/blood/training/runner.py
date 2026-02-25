"""Main training entry point for Sample Factory v2."""

import sys
import os
import signal
import atexit
import logging

import torch
import yaml
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner
from sample_factory.algo.learning.learner import Learner

from ..cfg import add_blood_args, blood_override_defaults
from ..env.blood_env import BloodMahjongEnv
from ..model.factory import register_blood_model
from .callbacks import BloodObserver
from .losses import BloodLossComputer

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


def _patch_learner():
    """Inject auxiliary + oracle distillation losses into the SF2 training loop.

    SF2 has no official custom-loss extension point, so we monkey-patch
    Learner._calculate_losses.  All loss logic lives in BloodLossComputer;
    this wrapper only handles minibatch pre-processing (adv std + adv clip).
    """
    _original = Learner._calculate_losses
    # Lazily initialized with cfg from the first Learner call so that
    # BloodLossComputer can read blood_metrics_interval from the config.
    _state = {"loss_computer": None}

    def _patched(self, mb, num_invalids):
        if _state["loss_computer"] is None:
            _state["loss_computer"] = BloodLossComputer(cfg=self.cfg)

        # Record raw advantage std before SF2 normalizes advantages in-place.
        raw_advantages = getattr(mb, "advantages", None)
        if raw_advantages is not None:
            self._last_raw_adv_std = float(raw_advantages.std().item())

        # Advantage clipping: prevent extreme samples from dominating gradients.
        adv_clip = getattr(self.cfg, "adv_clip", 0.0)
        if adv_clip > 0 and raw_advantages is not None:
            mb.advantages = torch.clamp(raw_advantages, -adv_clip, adv_clip)

        result = _original(self, mb, num_invalids)
        action_dist, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, summaries = result

        extra_loss, summaries = _state["loss_computer"].compute(
            self.actor_critic, mb, action_dist, value_loss, summaries,
            env_steps=getattr(self, "env_steps", 0),
        )

        # Add extra losses to value_loss so the PPO policy_loss curve stays clean.
        summaries["ppo_policy_loss"] = policy_loss.detach()
        summaries["extra_loss_total"] = extra_loss.squeeze().detach()
        value_loss = value_loss + extra_loss.squeeze()
        return action_dist, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, summaries

    Learner._calculate_losses = _patched


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


def run_training():
    _setup_process_cleanup()
    register_blood_components()
    _patch_learner()
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

    cfg, runner = make_runner(cfg)

    observer = BloodObserver(cfg)
    runner.register_observer(observer)

    runner.init()
    status = runner.run()
    return status


if __name__ == "__main__":
    sys.exit(run_training())
