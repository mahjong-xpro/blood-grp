"""Training callbacks for Blood Mahjong.

BloodObserver: unified AlgoObserver for league snapshots, oracle metrics,
and auxiliary task logging.  Also drives dynamic hyperparameter scheduling
via HyperparamScheduler.
"""

import logging
import os
from pathlib import Path

from sample_factory.algo.runners.runner import Runner, AlgoObserver

from blood.eval.elo import EloTracker
from blood.training.league import LeagueManager
from blood.training.scheduler import HyperparamScheduler

log = logging.getLogger(__name__)


class BloodObserver(AlgoObserver):
    """Unified AlgoObserver for Blood Mahjong training."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.league_enabled = getattr(cfg, "league_enabled", False)
        self._last_snapshot_step = 0
        self._runner = None

        # --- Elo tracker (optional, purely additive) ---
        elo_enabled = getattr(cfg, "blood_elo_enabled", True)
        self.elo_tracker = None
        if elo_enabled:
            # Persist Elo ratings alongside the experiment
            from sample_factory.utils.utils import experiment_dir
            try:
                exp_dir = experiment_dir(cfg=cfg)
            except Exception:
                exp_dir = "train_dir/blood_v2"
            elo_save_path = os.path.join(exp_dir, "elo_ratings.json")
            self.elo_tracker = EloTracker(
                save_path=elo_save_path,
                k_base=getattr(cfg, "blood_elo_k_base", 32.0),
                k_new_player=getattr(cfg, "blood_elo_k_new", 64.0),
                new_player_threshold=getattr(cfg, "blood_elo_new_threshold", 30),
            )
            log.info("[Elo] Tracker enabled, save_path=%s", elo_save_path)

        pool_dir = getattr(cfg, "league_pool_dir", "checkpoints/league")
        newest_weight = getattr(cfg, "league_newest_weight", 2.0)
        uniform_floor = getattr(cfg, "league_uniform_floor", 0.1)
        self_play_prob = getattr(cfg, "league_self_play_prob", 0.2)
        use_elo_sampling = getattr(cfg, "blood_elo_sampling", False)
        elo_sigma = getattr(cfg, "blood_elo_sampling_sigma", 200.0)
        self.league = LeagueManager(
            pool_dir,
            newest_weight=newest_weight,
            uniform_floor=uniform_floor,
            self_play_prob=self_play_prob,
            elo_tracker=self.elo_tracker,
            use_elo_sampling=use_elo_sampling,
            elo_sampling_sigma=elo_sigma,
        )
        self.league_add_every = getattr(cfg, "league_add_every", 50000)

        # Dynamic hyperparameter scheduler
        self._scheduler = HyperparamScheduler.from_config(cfg)
        if self._scheduler.schedules:
            log.info(
                "[Scheduler] Loaded %d schedule(s): %s",
                len(self._scheduler.schedules),
                ", ".join(s.param_name for s in self._scheduler.schedules),
            )

    def on_init(self, runner: Runner) -> None:
        self._runner = runner

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        policy_id = 0
        env_steps = runner.env_steps.get(policy_id, 0)

        # Apply dynamic hyperparameter schedules
        self._apply_schedules(runner, policy_id, env_steps)

        if not self.league_enabled:
            return

        if env_steps - self._last_snapshot_step >= self.league_add_every:
            self._snapshot_to_league(runner, policy_id, env_steps)
            self._last_snapshot_step = env_steps

    def _apply_schedules(self, runner: Runner, policy_id: int, env_steps: int) -> None:
        """Apply scheduled hyperparameter updates to the learner and cfg."""
        updates = self._scheduler.step(env_steps)
        if not updates:
            return

        learner_worker = runner.learners.get(policy_id)
        for param, value in updates.items():
            # Update on the learner's cfg (used by SF2 loss computation)
            if hasattr(self.cfg, param):
                setattr(self.cfg, param, value)
            # Also update on the learner object directly if it exists
            if learner_worker is not None and learner_worker.learner is not None:
                learner = learner_worker.learner
                if hasattr(learner.cfg, param):
                    setattr(learner.cfg, param, value)
            log.info("[Scheduler] %s = %.6f at step %d", param, value, env_steps)

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
        except FileNotFoundError:
            # SF2 rotated the checkpoint between our glob and copy — benign race.
            log.debug("SF2 checkpoint disappeared during league copy: %s", latest)
            save_path.unlink(missing_ok=True)
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
        if learner_worker is not None and learner_worker.learner is not None:
            ac = learner_worker.learner.actor_critic

        if ac is not None and hasattr(ac, "oracle_enabled") and ac.oracle_enabled:
            writer.add_scalar("blood/oracle_enabled", 1, env_steps)

        # Log raw advantage std (pre-normalization) from the learner's last minibatch.
        # SF2 records adv_std=0 because it logs post-normalization advantages.
        if learner_worker is not None and learner_worker.learner is not None:
            raw_adv_std = getattr(learner_worker.learner, "_last_raw_adv_std", None)
            if raw_adv_std is not None:
                writer.add_scalar("blood/raw_adv_std", float(raw_adv_std), env_steps)

        # Log current scheduler state
        sched_summary = self._scheduler.get_summary()
        for param, value in sched_summary.items():
            writer.add_scalar(f"blood/sched_{param}", value, env_steps)

        # --- Elo metrics ---
        self._log_elo_summaries(writer, env_steps)

    def _log_elo_summaries(self, writer, env_steps: int) -> None:
        """Log Elo-related metrics to TensorBoard."""
        if self.elo_tracker is None:
            return

        # Resolve the underlying TB writer (SF2 wraps it)
        tb = writer.writer if hasattr(writer, "writer") else writer

        # Current policy Elo
        current_elo = self.elo_tracker.get_rating("current_policy")
        tb.add_scalar("blood/elo_current", current_elo, env_steps)

        # Pool stats: best and mean Elo across league checkpoints
        leaderboard = self.elo_tracker.get_leaderboard(top_n=0)  # 0 → all
        if leaderboard:
            # Filter to league checkpoint entries only
            pool_entries = [
                (name, stats) for name, stats in leaderboard
                if name.startswith("league_ckpt_")
            ]
            if pool_entries:
                best_elo = pool_entries[0][1].elo  # already sorted desc
                mean_elo = sum(s.elo for _, s in pool_entries) / len(pool_entries)
                tb.add_scalar("blood/elo_best", best_elo, env_steps)
                tb.add_scalar("blood/elo_pool_mean", mean_elo, env_steps)
