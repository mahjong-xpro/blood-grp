"""联赛管理器：自博弈对手采样系统。

改进点：
- 使用文件名中的 env_steps 数字排序（而非 st_mtime，避免 NFS/容器环境不可靠）
- 多项式衰减 α=2.0 + uniform_floor 保底，提高有效多样性
- self_play_prob 支持当前策略 vs 自身对战
"""

import re
import shutil
import logging
from pathlib import Path

import numpy as np

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


class LeagueManager:
    """管理历史模型 checkpoint 池，用于自博弈对手采样。

    采样策略：多项式衰减 (rank^-alpha) + uniform_floor 保底概率，
    平衡新近偏好与多样性。支持 self_play_prob 概率返回 None
    表示使用当前最新策略自博弈。
    """

    def __init__(
        self,
        pool_dir: str,
        newest_weight: float = 2.0,       # α 从 3.0 降到 2.0，提高有效多样性
        max_pool_size: int = 50,
        uniform_floor: float = 0.1,       # 最低采样概率下限，确保旧 checkpoint 也能被采样
        self_play_prob: float = 0.2,      # 20% 概率使用当前策略自博弈
    ):
        self.pool_dir = Path(pool_dir)
        self.newest_weight = newest_weight
        self.max_pool_size = max_pool_size
        self.uniform_floor = uniform_floor
        self.self_play_prob = self_play_prob
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

    def sample_opponent(self):
        """采样一个对手 checkpoint。

        采样策略：
        1. 以 self_play_prob 概率返回 None（表示使用当前最新策略自博弈）
        2. 否则从历史池中按多项式衰减 + uniform_floor 采样

        权重公式: w(r) = (1 - floor) * [(1+r)^(-α) / Z] + floor * (1/n)
        其中 r=0 为最新，α=newest_weight，floor=uniform_floor。
        """
        # 自博弈：以 self_play_prob 概率使用当前策略
        if self._rng.random() < self.self_play_prob:
            log.debug("自博弈采样：使用当前最新策略")
            return None

        checkpoints = self.get_checkpoints()
        if not checkpoints:
            return None

        n = len(checkpoints)
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
        """池超出 max_pool_size 时移除最旧的 checkpoint。"""
        checkpoints = self.get_checkpoints()
        while len(checkpoints) > self.max_pool_size:
            oldest = checkpoints.pop()  # 列表末尾 = env_steps 最小 = 最旧
            oldest.unlink()
            log.info("淘汰最旧 checkpoint: %s", oldest.name)

    def pool_size(self) -> int:
        return len(self.get_checkpoints())

    def remove_checkpoint(self, path) -> None:
        """删除池中的 checkpoint（如损坏文件）。"""
        try:
            Path(path).unlink(missing_ok=True)
            log.warning("从联赛池移除损坏 checkpoint: %s", path)
        except Exception as e:
            log.warning("移除 checkpoint 失败 %s: %s", path, e)
