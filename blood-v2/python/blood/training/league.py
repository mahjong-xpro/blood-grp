"""League Manager for self-play opponent sampling."""

import shutil
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class LeagueManager:
    """Manages a pool of historical model checkpoints for self-play.

    Sampling uses polynomial decay (rank^-alpha) to balance recency bias with diversity.
    Automatically evicts oldest checkpoints when pool exceeds max size.
    """

    def __init__(
        self,
        pool_dir: str,
        newest_weight: float = 3.0,
        max_pool_size: int = 50,
    ):
        self.pool_dir = Path(pool_dir)
        self.newest_weight = newest_weight
        self.max_pool_size = max_pool_size
        self._rng = np.random.default_rng()

    def get_checkpoints(self):
        """Return list of checkpoint paths sorted by age (newest first)."""
        if not self.pool_dir.exists():
            return []
        checkpoints = sorted(
            self.pool_dir.glob("checkpoint_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return checkpoints

    def sample_opponent(self):
        """Sample an opponent checkpoint using polynomial decay.

        Weight for rank r (0 = newest): w(r) = 1 / (1 + r)^alpha
        where alpha = newest_weight. Avoids exponential degeneracy while
        still biasing towards newer models.
        """
        checkpoints = self.get_checkpoints()
        if not checkpoints:
            return None

        n = len(checkpoints)
        alpha = self.newest_weight
        weights = np.array([
            1.0 / (1.0 + rank) ** alpha for rank in range(n)
        ], dtype=np.float64)
        weights /= weights.sum()

        idx = self._rng.choice(n, p=weights)
        return checkpoints[idx]

    def add_checkpoint(self, source_path: Path):
        """Copy a checkpoint file into the pool directory, evicting oldest if at capacity."""
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        dest = self.pool_dir / source_path.name
        if not dest.exists():
            shutil.copy2(source_path, dest)
            log.info("Added checkpoint to league pool: %s", dest)

        self._evict_if_needed()

    def _evict_if_needed(self):
        """Remove oldest checkpoints if pool exceeds max size."""
        checkpoints = self.get_checkpoints()
        while len(checkpoints) > self.max_pool_size:
            oldest = checkpoints.pop()
            oldest.unlink()
            log.info("Evicted oldest checkpoint: %s", oldest.name)

    def pool_size(self) -> int:
        return len(self.get_checkpoints())

    def remove_checkpoint(self, path) -> None:
        """Delete a checkpoint from the pool (e.g. corrupted file)."""
        try:
            Path(path).unlink(missing_ok=True)
            log.warning("Removed corrupted checkpoint from league pool: %s", path)
        except Exception as e:
            log.warning("Failed to remove checkpoint %s: %s", path, e)
