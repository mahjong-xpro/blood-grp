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


def _inject_config_yaml():
    """Parse --config <yaml> from sys.argv, expand yaml keys into sys.argv,
    then remove --config so SF2 never sees it."""
    argv = sys.argv[1:]
    if "--config" not in argv:
        return
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        return
    yaml_path = argv[idx + 1]
    # Remove --config <path> from sys.argv
    sys.argv = [sys.argv[0]] + argv[:idx] + argv[idx + 2:]

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        return

    # Keys handled outside SF2 arg parsing — skip them here.
    # Includes: keys not registered as SF2 args, and bool-with-value args
    # that have dedicated --no_xxx counterparts (oracle_enabled, league_enabled).
    _skip = {"encoder_custom", "oracle_enabled", "league_enabled"}

    # SF2 args that use type=str2bool (accept --key False to disable).
    # store_true args cannot be set to False via CLI, so we skip them when False.
    _str2bool_args = {"use_rnn", "normalize_input", "normalize_returns"}

    # Append yaml values as CLI args (only if not already in sys.argv)
    existing = set(sys.argv)
    for key, val in cfg.items():
        if key in _skip:
            continue
        flag = f"--{key}"
        if flag in existing:
            continue
        if isinstance(val, bool):
            if val:
                sys.argv.append(flag)
            elif key in _str2bool_args:
                # str2bool args: pass --key False to override set_defaults(key=True)
                sys.argv.extend([flag, "False"])
            # else: store_true arg with False value — skip (can't unset via CLI)
        else:
            sys.argv.extend([flag, str(val)])


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

    SF2 training calls forward_head → forward_core → forward_tail (never forward()),
    so we compute extra losses here using the encoder output cached by forward_head.

    Losses injected:
    1. AuxHead: opponent dingque + waiting tiles prediction (CE + BCE)
    2. Oracle distillation: KL(student || oracle) using perfect-info teacher
       - Oracle CE is weighted by advantages to avoid circular dependency
       - Student distillation uses KL divergence against oracle (detached)
    """
    _original = Learner._calculate_losses

    def _patched(self, mb, num_invalids):
        # Record raw advantage std before SF2 normalizes advantages.
        # SF2 normalizes advantages in-place before calling _calculate_losses,
        # so we capture the std from the raw returns here for monitoring.
        raw_advantages = getattr(mb, "advantages", None)
        if raw_advantages is not None:
            self._last_raw_adv_std = float(raw_advantages.std().item())

        # Advantage clipping: clip to [-adv_clip, adv_clip] to prevent extreme
        # advantage samples (observed ±4.7 in warmup) from dominating gradients.
        adv_clip = getattr(self.cfg, "adv_clip", 0.0)
        if adv_clip > 0 and raw_advantages is not None:
            mb.advantages = torch.clamp(raw_advantages, -adv_clip, adv_clip)

        result = _original(self, mb, num_invalids)
        action_dist, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, summaries = result

        ac = self.actor_critic
        features = getattr(ac, "_cached_encoder_out", None)  # post-enc_proj (1024); used as forward-pass guard
        core_features = getattr(ac, "_cached_core_out", None)  # post-LSTM (1024); used by AuxHead
        obs = getattr(ac, "_cached_obs", None)

        if features is None or obs is None or not ac.training:
            return result

        cache_gen = getattr(ac, "_cache_gen", 0)
        loss_gen = getattr(ac, "_loss_gen", 0)
        if cache_gen <= loss_gen:
            log.warning("Stale encoder cache detected (gen %d <= %d); skipping aux losses", cache_gen, loss_gen)
            return result
        ac._loss_gen = cache_gen

        device = value_loss.device if hasattr(value_loss, 'device') else 'cpu'
        extra_loss = torch.zeros(1, device=device)

        if getattr(ac, "_aux_enabled", False):
            shanten_labels = obs.get("shanten_labels")
            ow_labels = obs.get("ow_labels")
            if shanten_labels is not None and ow_labels is not None and core_features is not None:
                aux_loss = ac.aux_head.loss(
                    core_features, shanten_labels, ow_labels,
                    shanten_weight=ac.shanten_weight, ow_weight=ac.ow_weight,
                )
                extra_loss = extra_loss + aux_loss
                summaries["aux_loss"] = aux_loss.detach()

        if getattr(ac, "oracle_enabled", False):
            oracle_obs = obs.get("oracle_obs")
            action_mask = obs.get("action_mask")
            if oracle_obs is not None:
                oracle_logits, oracle_values = ac.oracle_encoder(oracle_obs)

                student_logits = getattr(action_dist, 'logits', None)
                if student_logits is None:
                    log.warning("Cannot find 'logits' attribute on action_dist; skipping distillation")
                else:
                    mask_bool = action_mask.bool() if action_mask is not None else None
                    distill_loss = ac.distill_loss_fn(student_logits, oracle_logits.detach(), mask_bool)
                    extra_loss = extra_loss + ac.distill_weight * distill_loss
                    summaries["distill_loss"] = distill_loss.detach()

                oracle_ce_weight = getattr(ac, "oracle_ce_weight", 0.1)
                advantages = getattr(mb, "advantages", None)
                oracle_logits_masked = oracle_logits.clone()
                if action_mask is not None:
                    oracle_logits_masked = oracle_logits_masked.masked_fill(
                        ~action_mask.bool(), torch.finfo(oracle_logits.dtype).min
                    )
                oracle_ce_raw = torch.nn.functional.cross_entropy(
                    oracle_logits_masked,
                    mb.actions.long(),
                    reduction="none",
                )
                if advantages is not None:
                    adv_weights = torch.clamp(advantages.detach(), min=0.0)
                    adv_weights = adv_weights / (adv_weights.mean() + 1e-8)
                    oracle_ce = (oracle_ce_raw * adv_weights).mean()
                else:
                    oracle_ce = oracle_ce_raw.mean()
                extra_loss = extra_loss + oracle_ce_weight * oracle_ce
                summaries["oracle_ce"] = oracle_ce.detach()

                # Oracle value distillation: train student critic to match Oracle's
                # perfect-info value estimate. Oracle values are more accurate because
                # the Oracle sees all hands and the wall; this improves credit assignment.
                oracle_value_distill_weight = getattr(ac, "oracle_value_distill_weight", 0.5)
                if oracle_value_distill_weight > 0:
                    student_values = getattr(ac, "_cached_values", None)
                    if student_values is not None:
                        sv = student_values.view(-1)
                        ov = oracle_values.squeeze().detach().view(-1)
                        if sv.shape == ov.shape:
                            value_distill_loss = torch.nn.functional.mse_loss(sv, ov)
                            extra_loss = extra_loss + oracle_value_distill_weight * value_distill_loss
                            summaries["oracle_value_distill_loss"] = value_distill_loss.detach()

        # Add extra losses to value_loss so that the PPO policy_loss curve in
        # TensorBoard remains clean (pure PPO gradient signal).
        # summaries["ppo_policy_loss"] preserves the original for monitoring.
        # summaries["extra_loss_total"] shows the combined auxiliary signal.
        summaries["ppo_policy_loss"] = policy_loss.detach()
        summaries["extra_loss_total"] = extra_loss.squeeze().detach()
        value_loss = value_loss + extra_loss.squeeze()

        # Monitor logprob extremes: if max_abs_logprob keeps growing past ~8,
        # the policy is becoming degenerate (near-deterministic on some actions).
        student_logits = getattr(action_dist, 'logits', None)
        if student_logits is not None:
            log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
            summaries["blood/max_abs_logprob"] = log_probs.abs().max().detach()
            summaries["blood/mean_abs_logprob"] = log_probs.abs().mean().detach()

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
    _inject_config_yaml()

    # SF2 requires --env as a mandatory CLI argument before set_defaults can run.
    if "--env" not in sys.argv:
        sys.argv.extend(["--env", "blood_mahjong"])

    parser, partial_cfg = parse_sf_args(evaluation=False)
    add_blood_args(parser)
    blood_override_defaults(parser)
    cfg = parse_full_cfg(parser)

    cfg, runner = make_runner(cfg)

    observer = BloodObserver(cfg)
    runner.register_observer(observer)

    runner.init()
    status = runner.run()
    return status


if __name__ == "__main__":
    sys.exit(run_training())
