"""Population-Based Training (PBT) controller for Blood Mahjong.

Coordinates N parallel training instances, periodically evaluating,
selecting top performers, and mutating hyperparameters.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class PBTMember:
    """A single member of the PBT population."""
    member_id: int
    hyperparams: Dict[str, float]
    checkpoint_path: Optional[str] = None
    elo: float = 1500.0
    win_rate: float = 0.0
    env_steps: int = 0


# Hyperparameters that PBT can mutate, with (min, max) ranges
PBT_SEARCH_SPACE = {
    "exploration_loss_coeff": (0.005, 0.05),
    "oracle_distill_weight": (0.005, 0.1),
    "oracle_ce_weight": (0.01, 0.2),
    "reward_tsumo_bonus": (0.02, 0.2),
    "reward_deal_in_penalty": (0.01, 0.1),
    "reward_safe_discard": (0.0, 0.03),
    "reward_shanten_progress": (0.001, 0.01),
    "reward_rank_bonus": (0.05, 0.4),
    "ppo_clip_ratio": (0.1, 0.25),
    "learning_rate": (3e-5, 3e-4),
}


class PBTController:
    """Population-Based Training controller.

    Manages a population of parallel training runs, periodically:
    1. Evaluating each member via arena games
    2. Exploiting: bottom fraction copies top fraction's weights
    3. Exploring: mutating hyperparameters by perturb_factor
    """

    # PLACEHOLDER_METHODS

    def __init__(
        self,
        population_size: int = 4,
        eval_every: int = 1_000_000,
        exploit_fraction: float = 0.2,
        perturb_factor: float = 1.2,
        work_dir: str = "pbt_runs",
    ):
        self.population_size = population_size
        self.eval_every = eval_every
        self.exploit_fraction = exploit_fraction
        self.perturb_factor = perturb_factor
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.population: List[PBTMember] = []
        self._last_eval_steps: Dict[int, int] = {}

    def initialize_population(self, base_hyperparams: Dict[str, float]) -> List[PBTMember]:
        """Create initial population with perturbed hyperparameters."""
        import random

        self.population = []
        for i in range(self.population_size):
            hp = {}
            for key, value in base_hyperparams.items():
                if key in PBT_SEARCH_SPACE:
                    lo, hi = PBT_SEARCH_SPACE[key]
                    # Random perturbation within search space
                    factor = random.uniform(1.0 / self.perturb_factor, self.perturb_factor)
                    hp[key] = max(lo, min(hi, value * factor))
                else:
                    hp[key] = value

            member = PBTMember(member_id=i, hyperparams=hp)
            self.population.append(member)
            self._last_eval_steps[i] = 0

        self._save_state()
        return self.population

    def should_evaluate(self, member_id: int, current_steps: int) -> bool:
        """Check if a member is due for evaluation."""
        last = self._last_eval_steps.get(member_id, 0)
        return current_steps - last >= self.eval_every

    def update_member(self, member_id: int, elo: float, win_rate: float, env_steps: int,
                      checkpoint_path: str):
        """Update a member's evaluation results."""
        member = self.population[member_id]
        member.elo = elo
        member.win_rate = win_rate
        member.env_steps = env_steps
        member.checkpoint_path = checkpoint_path
        self._last_eval_steps[member_id] = env_steps

    def step(self) -> List[Dict]:
        """Run exploit + explore on the population.

        Returns list of actions: [{"member_id": int, "action": "copy"|"mutate", ...}]
        """
        actions = []
        n = len(self.population)
        if n < 2:
            return actions

        # Sort by Elo descending
        ranked = sorted(self.population, key=lambda m: m.elo, reverse=True)
        n_exploit = max(1, int(n * self.exploit_fraction))

        top_members = ranked[:n_exploit]
        bottom_members = ranked[-n_exploit:]

        import random

        for bottom in bottom_members:
            # Exploit: copy weights from a random top member
            donor = random.choice(top_members)
            if donor.checkpoint_path and donor.member_id != bottom.member_id:
                actions.append({
                    "member_id": bottom.member_id,
                    "action": "copy",
                    "from_member": donor.member_id,
                    "from_checkpoint": donor.checkpoint_path,
                })

                # Explore: mutate hyperparameters
                new_hp = {}
                for key, value in donor.hyperparams.items():
                    if key in PBT_SEARCH_SPACE:
                        lo, hi = PBT_SEARCH_SPACE[key]
                        factor = random.choice([
                            1.0 / self.perturb_factor,
                            1.0,
                            self.perturb_factor,
                        ])
                        new_hp[key] = max(lo, min(hi, value * factor))
                    else:
                        new_hp[key] = value

                bottom.hyperparams = new_hp
                actions.append({
                    "member_id": bottom.member_id,
                    "action": "mutate",
                    "new_hyperparams": new_hp,
                })

        self._save_state()
        return actions

    def get_member_config_overrides(self, member_id: int) -> Dict[str, float]:
        """Get config overrides for a specific member."""
        return dict(self.population[member_id].hyperparams)

    def _save_state(self):
        """Persist PBT state to disk atomically.

        Writes to a temporary file first, then atomically renames to prevent
        corruption if the process crashes mid-write.
        """
        state = {
            "population": [
                {
                    "member_id": m.member_id,
                    "hyperparams": m.hyperparams,
                    "elo": m.elo,
                    "win_rate": m.win_rate,
                    "env_steps": m.env_steps,
                    "checkpoint_path": m.checkpoint_path,
                }
                for m in self.population
            ],
            "last_eval_steps": self._last_eval_steps,
        }
        state_path = self.work_dir / "pbt_state.json"
        tmp_path = self.work_dir / "pbt_state.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(str(tmp_path), str(state_path))

    def load_state(self) -> bool:
        """Load PBT state from disk. Returns True if loaded successfully."""
        state_path = self.work_dir / "pbt_state.json"
        if not state_path.exists():
            return False
        try:
            with open(state_path) as f:
                state = json.load(f)
            self.population = [
                PBTMember(**m) for m in state["population"]
            ]
            self._last_eval_steps = {
                int(k): v for k, v in state["last_eval_steps"].items()
            }
            return True
        except Exception as e:
            log.warning("Failed to load PBT state: %s", e)
            return False
