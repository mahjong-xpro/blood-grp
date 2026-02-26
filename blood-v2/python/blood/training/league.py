"""联赛管理器：自博弈对手采样系统。

改进点：
- 使用文件名中的 env_steps 数字排序（而非 st_mtime，避免 NFS/容器环境不可靠）
- 多项式衰减 α=2.0 + uniform_floor 保底，提高有效多样性
- self_play_prob 支持当前策略 vs 自身对战
- 可选 Elo-weighted 采样：Gaussian 权重偏好 Elo 接近的对手
- 稀疏保留淘汰：池满时保留最新 50% 密集 + 旧 50% 稀疏采样，最大化时间跨度
- 冻结窗口：最近 N 个 checkpoint 不参与采样，避免与过于相似的策略对打
"""

from __future__ import annotations

import math
import re
import shutil
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from blood.eval.elo import EloTracker

log = logging.getLogger(__name__)

# 从 checkpoint 文件名提取 env_steps 的正则，格式: checkpoint_{env_steps}.pth
_CKPT_STEP_RE = re.compile(r"checkpoint_(\d+)\.pth$")


def _extract_env_steps(path: Path) -> int:
    """从 checkpoint 文件名提取 env_steps 数字，用于可靠排序。

    比 st_mtime 更可靠：NFS/容器环境中文件修改时间可能不准确。
    """
    m = _CKPT_STEP_RE.search(path.name)
    if m:
        return int(m.group(1))
    # 回退：无法解析时返回 0，排到最旧位置
    log.warning("无法从文件名提取 env_steps: %s，回退为 0", path.name)
    return 0


def _ckpt_name(path: Path) -> str:
    """Derive a stable Elo player name from a checkpoint path."""
    m = _CKPT_STEP_RE.search(path.name)
    if m:
        return f"league_ckpt_{m.group(1)}"
    return f"league_ckpt_{path.stem}"


class LeagueManager:
    """管理历史模型 checkpoint 池，用于自博弈对手采样。

    采样策略：多项式衰减 (rank^-alpha) + uniform_floor 保底概率，
    平衡新近偏好与多样性。支持 self_play_prob 概率返回 None
    表示使用当前最新策略自博弈。

    可选 Elo-weighted 采样：当 use_elo_sampling=True 且提供了 elo_tracker 时，
    使用 Gaussian 权重偏好 Elo 接近当前策略的对手，形成自然课程学习。
    """

    def __init__(
        self,
        pool_dir: str,
        newest_weight: float = 2.0,       # α 从 3.0 降到 2.0，提高有效多样性
        max_pool_size: int = 50,
        uniform_floor: float = 0.1,       # 最低采样概率下限，确保旧 checkpoint 也能被采样
        self_play_prob: float = 0.2,      # 20% 概率使用当前策略自博弈
        frozen_window: int = 0,           # 最近 N 个 checkpoint 不参与采样
        elo_tracker: Optional[EloTracker] = None,
        use_elo_sampling: bool = False,
        elo_sampling_sigma: float = 200.0,
    ):
        self.pool_dir = Path(pool_dir)
        self.newest_weight = newest_weight
        self.max_pool_size = max_pool_size
        self.uniform_floor = uniform_floor
        self.self_play_prob = self_play_prob
        self.frozen_window = frozen_window
        self._elo_tracker = elo_tracker
        self._use_elo_sampling = use_elo_sampling
        self._elo_sigma = elo_sampling_sigma
        self._rng = np.random.default_rng()

    def get_checkpoints(self):
        """返回 checkpoint 路径列表，按 env_steps 降序排列（最新在前）。

        使用文件名中的 env_steps 数字排序，比 st_mtime 更可靠。
        """
        if not self.pool_dir.exists():
            return []
        checkpoints = sorted(
            self.pool_dir.glob("checkpoint_*.pth"),
            key=_extract_env_steps,
            reverse=True,  # 最新（env_steps 最大）在前
        )
        return checkpoints

    def _elo_weights(self, current_elo: float, checkpoints: list[Path]) -> list[float]:
        """Compute Elo-based sampling weights (Gaussian around current rating).

        Prefers opponents within ±sigma Elo of the current policy, creating a
        natural curriculum: as the agent improves, it faces stronger opponents.
        """
        sigma = self._elo_sigma
        weights = []
        for ckpt in checkpoints:
            opp_elo = self._elo_tracker.get_rating(_ckpt_name(ckpt))
            diff = abs(current_elo - opp_elo)
            w = math.exp(-0.5 * (diff / sigma) ** 2)
            weights.append(max(w, 0.01))  # floor to ensure exploration
        return weights

    def sample_opponent(self, current_elo: Optional[float] = None):
        """采样一个对手 checkpoint。

        采样策略：
        1. 以 self_play_prob 概率返回 None（表示使用当前最新策略自博弈）
        2. 排除冻结窗口内的 checkpoint（最近 frozen_window 个不参与采样）
        3. 若 use_elo_sampling=True 且有 elo_tracker，使用 Elo-based Gaussian 权重
        4. 否则从历史池中按多项式衰减 + uniform_floor 采样

        权重公式 (poly): w(r) = (1 - floor) * [(1+r)^(-α) / Z] + floor * (1/n)
        其中 r=0 为最新，α=newest_weight，floor=uniform_floor。
        """
        # 自博弈：以 self_play_prob 概率使用当前策略
        if self._rng.random() < self.self_play_prob:
            log.debug("自博弈采样：使用当前最新策略")
            return None

        checkpoints = self.get_checkpoints()
        if not checkpoints:
            return None

        # 冻结窗口：排除最近 N 个 checkpoint，避免与过于相似的策略对打
        if self.frozen_window > 0 and len(checkpoints) > self.frozen_window:
            checkpoints = checkpoints[self.frozen_window:]
        elif self.frozen_window > 0 and len(checkpoints) <= self.frozen_window:
            # 池太小，所有 checkpoint 都在冻结窗口内，回退到无冻结
            pass

        n = len(checkpoints)

        # Elo-weighted sampling when enabled and tracker is available
        if (
            self._use_elo_sampling
            and self._elo_tracker is not None
            and current_elo is not None
        ):
            elo_w = self._elo_weights(current_elo, checkpoints)
            total = sum(elo_w)
            weights = np.array([w / total for w in elo_w], dtype=np.float64)
            idx = self._rng.choice(n, p=weights)
            return checkpoints[idx]

        # Fallback: polynomial decay sampling
        alpha = self.newest_weight

        # 多项式衰减权重
        poly_weights = np.array([
            1.0 / (1.0 + rank) ** alpha for rank in range(n)
        ], dtype=np.float64)
        poly_weights /= poly_weights.sum()

        # 混合均匀分布保底，确保旧 checkpoint 也有最低采样概率
        # final_w = (1 - floor) * poly_w + floor * uniform
        floor = self.uniform_floor
        uniform = np.full(n, 1.0 / n, dtype=np.float64)
        weights = (1.0 - floor) * poly_weights + floor * uniform
        weights /= weights.sum()  # 归一化（理论上已归一，防止浮点误差）

        idx = self._rng.choice(n, p=weights)
        return checkpoints[idx]

    def add_checkpoint(self, source_path: Path):
        """将 checkpoint 文件复制到池目录，超出容量时淘汰最旧的。"""
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        dest = self.pool_dir / source_path.name
        if not dest.exists():
            shutil.copy2(source_path, dest)
            log.info("添加 checkpoint 到联赛池: %s", dest)

        self._evict_if_needed()

    def _evict_if_needed(self):
        """稀疏保留淘汰：池满时保留最新 50% 密集 + 旧 50% 稀疏采样。

        策略：将 checkpoint 按时间分为两半
        - 新半部分（前 50%）：完整保留，保证近期策略密度
        - 旧半部分（后 50%）：每隔 sparse_interval 个保留一个 + 始终保留最旧
        这样可以在固定池大小下最大化有效时间跨度覆盖。
        """
        checkpoints = self.get_checkpoints()  # 按 env_steps 降序
        if len(checkpoints) <= self.max_pool_size:
            return

        n = len(checkpoints)
        # 保留最新 50% 密集
        dense_count = self.max_pool_size // 2
        keep = set(range(dense_count))  # indices to keep (newest first)

        # 旧半部分稀疏保留
        old_indices = list(range(dense_count, n))
        remaining_slots = self.max_pool_size - dense_count

        if remaining_slots > 0 and old_indices:
            # 始终保留最旧的 checkpoint（index n-1）
            keep.add(n - 1)
            remaining_slots -= 1

            if remaining_slots > 0 and len(old_indices) > 1:
                # 在旧半部分中均匀间隔选取
                sparse_interval = max(len(old_indices) // (remaining_slots + 1), 1)
                for i in range(0, len(old_indices), sparse_interval):
                    if len(keep) >= self.max_pool_size:
                        break
                    keep.add(old_indices[i])

        # 删除不在保留集中的 checkpoint
        for i in range(n):
            if i not in keep:
                checkpoints[i].unlink()
                log.info("稀疏淘汰 checkpoint: %s", checkpoints[i].name)

    def pool_size(self) -> int:
        return len(self.get_checkpoints())

    def remove_checkpoint(self, path) -> None:
        """删除池中的 checkpoint（如损坏文件）。"""
        try:
            Path(path).unlink(missing_ok=True)
            log.warning("从联赛池移除损坏 checkpoint: %s", path)
        except Exception as e:
            log.warning("移除 checkpoint 失败 %s: %s", path, e)
