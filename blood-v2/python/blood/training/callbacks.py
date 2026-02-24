"""Training callbacks for Blood Mahjong.

BloodObserver: unified AlgoObserver for league snapshots, oracle metrics,
and auxiliary task logging.
"""

import logging
from pathlib import Path

from sample_factory.algo.runners.runner import Runner, AlgoObserver

from blood.training.league import LeagueManager

log = logging.getLogger(__name__)


class BloodObserver(AlgoObserver):
    """Unified AlgoObserver for Blood Mahjong training."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.league_enabled = getattr(cfg, "league_enabled", False)
        self._last_snapshot_step = 0
        self._runner = None

        pool_dir = getattr(cfg, "league_pool_dir", "checkpoints/league")
        newest_weight = getattr(cfg, "league_newest_weight", 3.0)
        self.league = LeagueManager(pool_dir, newest_weight)
        self.league_add_every = getattr(cfg, "league_add_every", 50000)

    def on_init(self, runner: Runner) -> None:
        self._runner = runner

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        if not self.league_enabled:
            return

        policy_id = 0
        env_steps = runner.env_steps.get(policy_id, 0)

        if env_steps - self._last_snapshot_step >= self.league_add_every:
            self._snapshot_to_league(runner, policy_id, env_steps)
            self._last_snapshot_step = env_steps

    def _snapshot_to_league(self, runner: Runner, policy_id: int, env_steps: int):
        import glob
        import os
        import shutil
        from os.path import join
        from sample_factory.utils.utils import experiment_dir

        # LearnerWorker.learner is None in the main process (it lives in the worker
        # subprocess). Instead, copy the latest checkpoint SF2 already saved to disk.
        # Use abspath so that `latest` is always an absolute path — a relative path
        # would break if the working directory differs between the glob and the copy.
        ckpt_dir = os.path.abspath(join(experiment_dir(cfg=runner.cfg), f"checkpoint_p{policy_id}"))
        checkpoints = sorted(glob.glob(join(ckpt_dir, "checkpoint_*.pth")))
        if not checkpoints:
            # Expected at training start: SF2 hasn't saved its first checkpoint yet.
            # Downgraded to debug to avoid log spam during the initial warm-up period.
            log.debug("No SF2 checkpoints found in %s; skipping league snapshot", ckpt_dir)
            return

        latest = checkpoints[-1]
        if not os.path.exists(latest):
            log.debug("SF2 checkpoint disappeared before league copy: %s", latest)
            return
        self.league.pool_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.league.pool_dir / f"checkpoint_{env_steps}.pth"

        try:
            shutil.copy2(latest, str(save_path))
            # Verify the copy is a valid PyTorch checkpoint (not a partial write).
            # weights_only=False needed: SF2 checkpoints contain numpy scalars.
            import torch
            torch.load(str(save_path), map_location="cpu", weights_only=False)
            self.league._evict_if_needed()
            log.info("Saved league checkpoint: %s (pool size: %d)",
                     save_path, self.league.pool_size())
        except Exception as e:
            log.warning("Failed to save league checkpoint: %s", e)
            # Remove corrupt file so it doesn't pollute the pool.
            try:
                save_path.unlink(missing_ok=True)
            except Exception:
                pass

    def extra_summaries(self, runner: Runner, policy_id, writer, env_steps: int) -> None:
        # Write league_pool_size as a plain integer directly to the SummaryWriter,
        # bypassing SF2's running-mean normalizer which would corrupt integer scalars.
        pool_size = float(self.league.pool_size())
        if hasattr(writer, "writer"):
            # SF2 wraps SummaryWriter; access the underlying TB writer directly
            writer.writer.add_scalar("blood/league_pool_size", pool_size, env_steps)
        else:
            writer.add_scalar("blood/league_pool_size", pool_size, env_steps)

        ac = None
        learner_worker = runner.learners.get(policy_id)
        if learner_worker is not None:
            ac = learner_worker.learner.actor_critic

        if ac is not None and hasattr(ac, "oracle_enabled") and ac.oracle_enabled:
            writer.add_scalar("blood/oracle_enabled", 1, env_steps)

        # Log raw advantage std (pre-normalization) from the learner's last minibatch.
        # SF2 records adv_std=0 because it logs post-normalization advantages.
        if learner_worker is not None:
            raw_adv_std = getattr(learner_worker.learner, "_last_raw_adv_std", None)
            if raw_adv_std is not None:
                writer.add_scalar("blood/raw_adv_std", float(raw_adv_std), env_steps)
