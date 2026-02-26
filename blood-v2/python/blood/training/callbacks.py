"""Training callbacks for Blood Mahjong.

BloodObserver: unified AlgoObserver for league snapshots, oracle metrics,
and auxiliary task logging.  Also drives dynamic hyperparameter scheduling
via HyperparamScheduler.

Includes periodic arena evaluation to update Elo ratings during training.
"""

import glob
import logging
import os
import threading
import time
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
            max_pool_size=getattr(cfg, "league_max_pool_size", 200),
            uniform_floor=uniform_floor,
            self_play_prob=self_play_prob,
            frozen_window=getattr(cfg, "league_frozen_window", 0),
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

        # --- Arena evaluation for Elo updates ---
        self._arena_eval_every = getattr(cfg, "blood_arena_eval_every", 0)
        self._arena_eval_games = getattr(cfg, "blood_arena_eval_games", 50)
        self._arena_eval_temperature = getattr(cfg, "blood_arena_eval_temperature", 0.1)
        self._last_arena_eval_step = 0
        self._arena_eval_running = False  # guard against concurrent evals
        self._arena_eval_lock = threading.Lock()
        # Latest arena result (written by bg thread, read by extra_summaries)
        self._arena_result = None
        self._arena_result_step = 0
        if self._arena_eval_every > 0 and self.elo_tracker is not None:
            log.info(
                "[Arena] Periodic evaluation enabled: every %d steps, %d games, temp=%.2f",
                self._arena_eval_every, self._arena_eval_games, self._arena_eval_temperature,
            )

    def on_init(self, runner: Runner) -> None:
        self._runner = runner

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        policy_id = 0
        env_steps = runner.env_steps.get(policy_id, 0)

        # Apply dynamic hyperparameter schedules
        self._apply_schedules(runner, policy_id, env_steps)

        # Periodic arena evaluation for Elo updates
        if (
            self._arena_eval_every > 0
            and self.elo_tracker is not None
            and env_steps - self._last_arena_eval_step >= self._arena_eval_every
        ):
            self._maybe_start_arena_eval(runner, policy_id, env_steps)
            self._last_arena_eval_step = env_steps

        if not self.league_enabled:
            return

        if env_steps - self._last_snapshot_step >= self.league_add_every:
            self._snapshot_to_league(runner, policy_id, env_steps)
            self._last_snapshot_step = env_steps

    def _apply_schedules(self, runner: Runner, policy_id: int, env_steps: int) -> None:
        """Log scheduled hyperparameter values (observer-side, logging only).

        Actual application happens inside the Learner process via the
        monkey-patched _calculate_losses (see runner.py _patch_learner).
        This method mirrors the computation for logging only.
        NOTE: Observer-side cfg updates are intentionally removed to prevent
        divergence between observer and learner cfg objects (Issue #R4-H6).
        """
        updates = self._scheduler.step(env_steps)
        if not updates:
            return

        # Entropy floor safety net (mirror the Learner-side logic for logging)
        entropy_floor = getattr(self.cfg, "blood_entropy_floor", 0.0)
        if entropy_floor > 0 and "exploration_loss_coeff" in updates:
            if updates["exploration_loss_coeff"] < entropy_floor:
                log.warning(
                    "[Scheduler] entropy %.6f < floor %.6f, clamping",
                    updates["exploration_loss_coeff"], entropy_floor,
                )
                updates["exploration_loss_coeff"] = entropy_floor

        for param, value in updates.items():
            log.info("[Scheduler] %s = %.6f at step %d", param, value, env_steps)

    def _find_latest_checkpoint(self, runner: Runner, policy_id: int) -> str | None:
        """Find the latest SF2 checkpoint on disk. Returns path or None."""
        from sample_factory.utils.utils import experiment_dir
        ckpt_dir = os.path.abspath(
            os.path.join(experiment_dir(cfg=runner.cfg), f"checkpoint_p{policy_id}")
        )
        checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pth")))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        return latest if os.path.exists(latest) else None

    def _maybe_start_arena_eval(self, runner: Runner, policy_id: int, env_steps: int) -> None:
        """Launch a background arena evaluation if one isn't already running."""
        with self._arena_eval_lock:
            if self._arena_eval_running:
                log.debug("[Arena] Skipping eval at step %d — previous eval still running", env_steps)
                return
            self._arena_eval_running = True

        ckpt_path = self._find_latest_checkpoint(runner, policy_id)
        if ckpt_path is None:
            log.debug("[Arena] No checkpoint available yet; skipping eval at step %d", env_steps)
            with self._arena_eval_lock:
                self._arena_eval_running = False
            return

        log.info("[Arena] Starting evaluation at step %d (%d games vs RuleBot)", env_steps, self._arena_eval_games)
        t = threading.Thread(
            target=self._run_arena_eval,
            args=(ckpt_path, env_steps),
            daemon=True,
            name=f"arena-eval-{env_steps}",
        )
        t.start()

    def _run_arena_eval(self, ckpt_path: str, env_steps: int) -> None:
        """Run arena evaluation in a background thread. Updates EloTracker."""
        try:
            from blood.env.blood_env import BloodMahjongEnv
            from blood.eval.arena import Arena
            from blood.model.inference import PolicyModel
            from blood.eval.evaluate import NeuralAgent

            t0 = time.time()
            model = PolicyModel.from_sf2_checkpoint(ckpt_path, device="cpu")
            agent = NeuralAgent(model, device="cpu", temperature=self._arena_eval_temperature)

            arena = Arena(
                BloodMahjongEnv,
                agent,
                baseline_mode="rulebot",
                elo_tracker=self.elo_tracker,
            )
            result = arena.evaluate(
                num_games=self._arena_eval_games,
                seed=env_steps,  # deterministic per step
                agent_name="current_policy",
            )
            elapsed = time.time() - t0

            # Store result for TensorBoard logging in extra_summaries
            # Use lock to ensure _arena_result and _arena_result_step are
            # read atomically by extra_summaries (Issue #R7-H4).
            with self._arena_eval_lock:
                self._arena_result = result
                self._arena_result_step = env_steps

            log.info(
                "[Arena] Step %d: win_rate=%.3f, avg_rank=%.2f, avg_score=%.0f, "
                "elo=%.1f (%d games in %.1fs)",
                env_steps, result.win_rate, result.avg_rank, result.avg_score,
                result.agent_elo, result.num_games, elapsed,
            )
        except Exception:
            log.exception("[Arena] Evaluation failed at step %d", env_steps)
        finally:
            with self._arena_eval_lock:
                self._arena_eval_running = False

    def _snapshot_to_league(self, runner: Runner, policy_id: int, env_steps: int):
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

        # Current policy Elo (updated by arena evaluation)
        current_elo = self.elo_tracker.get_rating("current_policy")
        tb.add_scalar("blood/elo_current", current_elo, env_steps)

        # Current policy game count (shows arena eval is working)
        current_stats = self.elo_tracker.get_stats("current_policy")
        if current_stats is not None:
            tb.add_scalar("blood/elo_games", float(current_stats.games), env_steps)

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

        # Arena evaluation metrics (from latest background eval)
        with self._arena_eval_lock:
            result = self._arena_result
        if result is not None and result.num_games > 0:
            tb.add_scalar("blood/arena_win_rate", result.win_rate, env_steps)
            tb.add_scalar("blood/arena_avg_rank", result.avg_rank, env_steps)
            tb.add_scalar("blood/arena_avg_score", result.avg_score, env_steps)
