"""
Learner patch that survives multiprocessing.spawn.

This module patches Learner methods at import time, ensuring patches are
applied in both the main process and spawned worker processes.

The key insight: multiprocessing.spawn re-imports all modules, so module-level
code (executed at import time) will run in every process. By placing patches
at module level instead of in a function called from main(), we ensure they're
applied everywhere.

This enables parallel training mode (10-15x speedup) while keeping all custom
losses (Oracle distillation, auxiliary tasks, etc.).
"""

import logging
from sample_factory.algo.learning.learner import Learner

log = logging.getLogger(__name__)

# Store original methods before patching
_original_calculate_losses = Learner._calculate_losses
_original_load_state = Learner._load_state

# Lazy-initialized state (per-process)
_loss_computer = None
_scheduler = None


def _get_loss_computer(cfg):
    """Get or create BloodLossComputer for this process."""
    global _loss_computer
    if _loss_computer is None:
        from blood.training.losses import BloodLossComputer
        _loss_computer = BloodLossComputer(cfg=cfg)
        log.info("[LearnerPatch] BloodLossComputer initialized in process %d", 
                 __import__('os').getpid())
    return _loss_computer


def _get_scheduler(cfg):
    """Get or create HyperparamScheduler for this process."""
    global _scheduler
    if _scheduler is None:
        from blood.training.scheduler import HyperparamScheduler
        _scheduler = HyperparamScheduler.from_config(cfg)
        if _scheduler.schedules:
            log.info("[LearnerPatch] HyperparamScheduler initialized with %d schedule(s)",
                     len(_scheduler.schedules))
    return _scheduler


def _patched_calculate_losses(self, mb, num_invalids):
    """Patched version of Learner._calculate_losses with Blood custom losses."""
    import torch
    
    # Get or create per-process singletons
    loss_computer = _get_loss_computer(self.cfg)
    scheduler = _get_scheduler(self.cfg)
    
    # Apply scheduled hyperparameter updates
    env_steps = getattr(self, "env_steps", 0)
    sched_updates = scheduler.step(env_steps)
    entropy_floor = getattr(self.cfg, "blood_entropy_floor", 0.0)
    if entropy_floor > 0 and "exploration_loss_coeff" in sched_updates:
        if sched_updates["exploration_loss_coeff"] < entropy_floor:
            sched_updates["exploration_loss_coeff"] = entropy_floor
    for param, val in sched_updates.items():
        if hasattr(self.cfg, param):
            setattr(self.cfg, param, val)
    
    # Record raw advantage std before SF2 normalizes
    raw_advantages = getattr(mb, "advantages", None)
    if raw_advantages is not None:
        self._last_raw_adv_std = float(raw_advantages.std().item())
    
    # Advantage clipping
    adv_clip = getattr(self.cfg, "adv_clip", 0.0)
    if adv_clip > 0 and raw_advantages is not None:
        mb.advantages = torch.clamp(raw_advantages, -adv_clip, adv_clip)
    
    # Call original SF2 loss calculation
    result = _original_calculate_losses(self, mb, num_invalids)
    action_dist, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, summaries = result
    
    # Compute Blood custom losses
    extra_loss, summaries = loss_computer.compute(
        self.actor_critic, mb, action_dist, value_loss, summaries,
        env_steps=env_steps,
    )
    
    # Add extra losses to policy_loss (keeps value_loss clean for diagnostics)
    summaries["ppo_policy_loss"] = policy_loss.detach()
    summaries["extra_loss_total"] = extra_loss.squeeze().detach()
    policy_loss = policy_loss + extra_loss.squeeze()
    
    return action_dist, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, summaries


def _patched_load_state(self, checkpoint_dict, load_progress=True):
    """Patched version of Learner._load_state with strict=False for cross-phase loading."""
    # Check if this is a cross-phase seed checkpoint (has marker key).
    # Using a marker key instead of checking init_checkpoint_path ensures
    # that normal --resume loads use strict=True.
    is_cross_phase = checkpoint_dict.get("_cross_phase_seed", False)
    if is_cross_phase:
        # Use strict=False for cross-phase loading
        model_sd = checkpoint_dict.get("model", {})
        if model_sd:
            missing, unexpected = self.actor_critic.load_state_dict(model_sd, strict=False)
            if missing:
                log.info("Cross-phase load: %d missing keys (new layers, randomly initialized)", len(missing))
                for k in missing[:5]:
                    log.info("  missing: %s", k)
                if len(missing) > 5:
                    log.info("  ... and %d more", len(missing) - 5)
            if unexpected:
                log.info("Cross-phase load: %d unexpected keys (skipped)", len(unexpected))
                for k in unexpected[:5]:
                    log.info("  unexpected: %s", k)
            loaded = len(model_sd) - len(unexpected)
            log.info("Cross-phase load: transferred %d/%d weights", loaded, len(model_sd))

            # Load optimizer state
            opt_state = checkpoint_dict.get("optimizer")
            if opt_state is not None:
                try:
                    self.optimizer.load_state_dict(opt_state)
                    log.info("Cross-phase load: optimizer state loaded (LR reset preserved)")
                except Exception as e:
                    log.warning("Cross-phase load: optimizer load failed (%s), using fresh optimizer", e)

            # Reset counters so the new phase starts from step 0
            if load_progress:
                self.train_step = checkpoint_dict.get("train_step", 0)
                self.env_steps = checkpoint_dict.get("env_steps", 0)
                log.info("Loaded training progress: train_step=%d, env_steps=%d",
                         self.train_step, self.env_steps)
            return
    # Default: use original strict loading
    return _original_load_state(self, checkpoint_dict, load_progress)


# Apply patches at module import time (runs in every process)
Learner._calculate_losses = _patched_calculate_losses
Learner._load_state = _patched_load_state
log.info("[LearnerPatch] Learner methods patched at import time (pid=%d)",
         __import__('os').getpid())