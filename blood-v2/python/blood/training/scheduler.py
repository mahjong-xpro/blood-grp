"""Dynamic hyperparameter scheduling within a training stage.

Supports linear, cosine, cyclic, and step schedules for any numeric
hyperparameter. Schedules are defined in YAML config and applied
via BloodObserver.on_training_step().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScheduleConfig:
    """Configuration for a single hyperparameter schedule."""

    param_name: str  # e.g., "exploration_loss_coeff"
    schedule_type: str  # "linear", "cosine", "cyclic", "step"
    start_value: float
    end_value: float
    start_step: int = 0
    end_step: int = 1_000_000
    # For cyclic schedule
    cycle_steps: int = 500_000
    cycle_min: float = 0.0
    cycle_max: float = 1.0
    # For step schedule
    milestones: list[int] = field(default_factory=list)  # step thresholds
    values: list[float] = field(default_factory=list)  # value at each milestone


class HyperparamScheduler:
    """Manages dynamic hyperparameter schedules during training.

    Usage::

        scheduler = HyperparamScheduler.from_config(cfg)
        # In on_training_step callback:
        updates = scheduler.step(env_steps)
        for param, value in updates.items():
            setattr(learner, param, value)
    """

    def __init__(self, schedules: list[ScheduleConfig]):
        self.schedules = schedules
        self._last_values: dict[str, float] = {}

    @classmethod
    def from_config(cls, cfg) -> HyperparamScheduler:
        """Parse schedule configs from the training config.

        Expects cfg attributes like::

            blood_schedule_entropy: "cosine,0.05,0.01,0,2000000"
            blood_schedule_adv_clip: "linear,5.0,3.0,10000000,50000000"
        """
        schedules: list[ScheduleConfig] = []

        # Parse entropy schedule
        raw = getattr(cfg, "blood_schedule_entropy", "")
        if raw:
            schedules.append(cls._parse_schedule("exploration_loss_coeff", raw))

        # Parse adv_clip schedule
        raw = getattr(cfg, "blood_schedule_adv_clip", "")
        if raw:
            schedules.append(cls._parse_schedule("adv_clip", raw))

        # Parse any generic schedules (semicolon-separated, colon between name and spec)
        raw = getattr(cfg, "blood_schedule_extra", "")
        if raw:
            for entry in raw.split(";"):
                parts = entry.strip().split(":")
                if len(parts) == 2:
                    schedules.append(cls._parse_schedule(parts[0], parts[1]))

        return cls(schedules)

    @staticmethod
    def _parse_schedule(param_name: str, raw: str) -> ScheduleConfig:
        """Parse a schedule string like ``'cosine,0.05,0.01,0,2000000'``."""
        parts = raw.split(",")
        stype = parts[0].strip()

        if stype in ("linear", "cosine"):
            return ScheduleConfig(
                param_name=param_name,
                schedule_type=stype,
                start_value=float(parts[1]),
                end_value=float(parts[2]),
                start_step=int(parts[3]) if len(parts) > 3 else 0,
                end_step=int(parts[4]) if len(parts) > 4 else 1_000_000,
            )
        elif stype == "cyclic":
            return ScheduleConfig(
                param_name=param_name,
                schedule_type="cyclic",
                start_value=float(parts[1]),
                end_value=float(parts[2]),
                cycle_steps=int(parts[3]) if len(parts) > 3 else 500_000,
                cycle_min=float(parts[1]),
                cycle_max=float(parts[2]),
            )
        elif stype == "step":
            milestones = [int(x) for x in parts[1::2]]
            values = [float(x) for x in parts[2::2]]
            if len(milestones) != len(values):
                raise ValueError(
                    f"Step schedule for '{param_name}' has {len(milestones)} "
                    f"milestones but {len(values)} values. Each milestone needs "
                    f"a corresponding value: 'step,M1,V1,M2,V2,...'"
                )
            return ScheduleConfig(
                param_name=param_name,
                schedule_type="step",
                start_value=values[0] if values else 0.0,
                end_value=values[-1] if values else 0.0,
                milestones=milestones,
                values=values,
            )
        else:
            raise ValueError(f"Unknown schedule type: {stype}")

    def step(self, env_steps: int) -> dict[str, float]:
        """Compute current values for all scheduled hyperparameters.

        Returns dict of ``{param_name: value}`` only for params that changed.
        """
        updates: dict[str, float] = {}
        for sched in self.schedules:
            value = self._compute_value(sched, env_steps)
            # Only report if changed (with small epsilon for float comparison)
            if (
                sched.param_name not in self._last_values
                or abs(self._last_values[sched.param_name] - value) > 1e-8
            ):
                updates[sched.param_name] = value
                self._last_values[sched.param_name] = value
        return updates

    @staticmethod
    def _compute_value(sched: ScheduleConfig, env_steps: int) -> float:
        """Compute the scheduled value at the given step."""
        if sched.schedule_type == "linear":
            if env_steps <= sched.start_step:
                return sched.start_value
            if env_steps >= sched.end_step:
                return sched.end_value
            progress = (env_steps - sched.start_step) / max(
                sched.end_step - sched.start_step, 1
            )
            return sched.start_value + (sched.end_value - sched.start_value) * progress

        elif sched.schedule_type == "cosine":
            if env_steps <= sched.start_step:
                return sched.start_value
            if env_steps >= sched.end_step:
                return sched.end_value
            progress = (env_steps - sched.start_step) / max(
                sched.end_step - sched.start_step, 1
            )
            # Cosine annealing: starts at start_value, ends at end_value
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return sched.end_value + (sched.start_value - sched.end_value) * cosine_factor

        elif sched.schedule_type == "cyclic":
            # Triangular wave between cycle_min and cycle_max
            phase = (env_steps % sched.cycle_steps) / max(sched.cycle_steps, 1)
            if phase < 0.5:
                return sched.cycle_min + (sched.cycle_max - sched.cycle_min) * (phase * 2)
            else:
                return sched.cycle_max - (sched.cycle_max - sched.cycle_min) * (
                    (phase - 0.5) * 2
                )

        elif sched.schedule_type == "step":
            # Find the last milestone <= env_steps
            value = sched.values[0] if sched.values else sched.start_value
            for ms, val in zip(sched.milestones, sched.values):
                if env_steps >= ms:
                    value = val
                else:
                    break
            return value

        return sched.start_value

    def get_summary(self) -> dict[str, Any]:
        """Return current state for logging."""
        return dict(self._last_values)
