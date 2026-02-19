import random
import torch
import numpy as np
import logging
from torch.utils.data import IterableDataset

# PERF-03: Numba JIT for sequential loops (optional, graceful fallback)
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):
        """No-op decorator when numba is not installed."""
        def _wrap(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _wrap

# Ensure libblood is initialized before importing modules that depend on it
# This is necessary when using 'spawn' multiprocessing method
try:
    import prelude  # This will initialize libblood
except ImportError:
    # Fallback: try to import libblood directly
    try:
        import libblood_loader
    except ImportError:
        pass
    try:
        import libblood
    except ImportError:
        import sys
        import os
        # If libblood is not found, try to add the project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import libblood

# from model import GRP
from reward_calculator import RewardCalculator
from libblood.dataset import GameplayLoader
from config import config


@_njit(cache=True)
def _td_lambda_inner(kyoku_rewards, at_kyoku, dones, apply_gamma, gamma, lambda_, game_size):
    """Numba-accelerated (or pure Python fallback) reverse TD(λ) loop."""
    td_returns = np.zeros(game_size, dtype=np.float64)
    running_return = 0.0
    for i in range(game_size - 1, -1, -1):
        k = at_kyoku[i]
        if dones[i]:
            running_return = kyoku_rewards[k]
        elif apply_gamma[i]:
            running_return = gamma * lambda_ * running_return
        # else: non-advancing step, running_return unchanged
        td_returns[i] = running_return
    return td_returns


@_njit(cache=True)
def _steps_to_done_inner(dones, apply_gamma, game_size):
    """Numba-accelerated (or pure Python fallback) reverse steps_to_done loop."""
    out = np.zeros(game_size, dtype=np.int64)
    steps = 0
    for i in range(game_size - 1, -1, -1):
        if dones[i]:
            steps = 0
        else:
            steps += apply_gamma[i]  # bool → 0/1
        out[i] = steps
    return out


def compute_td_lambda_returns(
    kyoku_rewards: np.ndarray,
    at_kyoku: np.ndarray,
    dones: np.ndarray,
    apply_gamma: np.ndarray,
    gamma: float,
    lambda_: float,
    game_size: int
) -> np.ndarray:
    """
    计算 TD(λ) 回报.

    标准 TD(λ):
        G_t^λ = r_t + γ * [(1-λ)*V(s_{t+1}) + λ*G_{t+1}^λ]

    当 target_network.enabled=false 时，无 V(s') bootstrap，公式退化为:
        G_t = r_t + γ*λ*G_{t+1}
    有效每步折扣 = γ×λ (而非 γ). λ<1 不提供方差缩减, 仅额外衰减信号.
    因此无 target network 时 λ 必须设为 1.0 (等价于 MC 回报).

    当 target_network.enabled=true 时，1-step TD 在 train.py 中计算，
    本函数仍用于计算 MC 回报作为参考/回退。
    """
    if lambda_ < 1.0:
        tn_enabled = config.get('target_network', {}).get('enabled', False)
        if not tn_enabled:
            import warnings
            eff = gamma * lambda_
            warnings.warn(
                f"td_lambda={lambda_} < 1.0 但 target_network 未启用. "
                f"有效折扣 γ_eff={eff:.4f} (非配置的 γ={gamma}). "
                f"请设 td_lambda=1.0 或启用 target_network.",
                stacklevel=2,
            )
    return _td_lambda_inner(kyoku_rewards, at_kyoku, dones, apply_gamma, gamma, lambda_, game_size)


class FileDatasetsIter(IterableDataset):
    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
    ):
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None

    def build_iter(self):
        # do not put it in __init__, it won't work on Windows
        # self.grp = GRP(**config['grp']['network'])
        # grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        # self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(config)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        try:
            data = self.loader.load_gz_log_files(file_list)
        except Exception as e:
            # Dataset construction must be replayable and self-consistent:
            # - labels must always be allowed by the computed action mask
            # - ding_que / kan_select windows must be legal
            #
            # If libblood reports a "Dataset mismatch" / "Mask mismatch", do NOT hide it by skipping;
            # fail fast so we can fix the underlying replay/labeling bug.
            msg = str(e)
            logging.error(f"Failed to load game logs from {len(file_list)} files: {msg}")
            if ("Dataset mismatch" in msg) or ("Mask mismatch" in msg):
                raise
            # For other issues (corrupt gz / parse errors), keep current behavior and skip this batch.
            logging.warning("Skipping this file batch due to non-fatal load error.")
            return
        
        for file in data:
            for game in file:
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku_raw = game.take_at_kyoku()
                # PyO3 returns Vec<u8> as Python bytes; convert to list of ints
                at_kyoku = list(at_kyoku_raw) if isinstance(at_kyoku_raw, bytes) else at_kyoku_raw
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()
                opponent_waits = game.take_opponent_waits()

                # per game
                game_score = game.take_game_score()
                player_id = game.take_player_id()

                game_size = len(obs)

                # GameScore provides scores_history as list of lists (kyoku x 4)
                # Convert to numpy array for easier slicing
                scores_history_list = game_score.take_scores_history()
                scores_history = np.array(scores_history_list, dtype=np.float64) # Float for division if needed, but int is fine. 
                # actually calc_delta_points expects numpy array to slice.
                
                rank_by_player = game_score.take_rank_by_player()
                final_scores = game_score.take_final_scores()

                # SBR Score-based (Maximize Points):
                # Scale: 1.0 reward = 10000 points
                kyoku_rewards = self.reward_calc.calc_delta_points(player_id, scores_history, final_scores) / 10000.0
                
                # 添加排名奖励到最后一个 kyoku (游戏结束时)
                if self.reward_calc.rank_bonus_enabled and len(kyoku_rewards) > 0:
                    rank_bonus = self.reward_calc.calc_rank_bonus(player_id, final_scores)
                    kyoku_rewards[-1] += rank_bonus
                
                # 添加动作级奖励 (和牌奖励 + 放铳惩罚)
                if self.reward_calc.action_bonus_enabled:
                    agari_count_list = game_score.take_agari_count()
                    houjuu_count_list = game_score.take_houjuu_count()
                    if len(agari_count_list) > 0 and len(houjuu_count_list) > 0:
                        agari_arr = np.array([row[player_id] for row in agari_count_list], dtype=np.float64)
                        houjuu_arr = np.array([row[player_id] for row in houjuu_count_list], dtype=np.float64)
                        for k in range(min(len(kyoku_rewards), len(agari_arr))):
                            action_bonus = self.reward_calc.calc_action_bonus(agari_arr[k], houjuu_arr[k])
                            kyoku_rewards[k] += action_bonus

                # Per-step Ding Que auxiliary bonus (only at the step where player chose DingQue).
                # Action indices 31=Man, 32=Pin, 33=Sou. See docs/DING_QUE_AUXILIARY_LEARNING.md.
                ding_que_aux_enabled = config.get('aux', {}).get('ding_que_aux_enabled', False)
                ding_que_aux_scale = config.get('aux', {}).get('ding_que_aux_scale', 0.02)
                ding_que_quality = game_score.take_ding_que_quality()
                ding_que_best_suit_list = game_score.take_ding_que_best_suit()
                if ding_que_aux_enabled and len(ding_que_quality) > 0:
                    ding_que_quality_arr = np.array(ding_que_quality, dtype=np.float64)
                    player_dq_quality = ding_que_quality_arr[:, player_id]
                else:
                    player_dq_quality = None
                # Rust Vec<[u8; 4]> is exposed as list of 4-byte sequences (bytes or list); take player_id-th column
                if len(ding_que_best_suit_list) > 0:
                    player_dq_best_suit = np.array(
                        [row[player_id] for row in ding_que_best_suit_list],
                        dtype=np.int64,
                    )
                else:
                    player_dq_best_suit = None

                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                scores_seq = np.concatenate((scores_history, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                # PERF-03: steps_to_done 使用 Numba 加速（或纯 Python fallback）
                dones_arr_std = np.asarray(dones, dtype=np.bool_)
                apply_gamma_arr_std = np.asarray(apply_gamma, dtype=np.int64)
                steps_to_done = _steps_to_done_inner(dones_arr_std, apply_gamma_arr_std, game_size)

                # TD(λ) 回报计算
                td_lambda_enabled = config.get('env', {}).get('td_lambda_enabled', False)
                td_lambda = config.get('env', {}).get('td_lambda', 0.95)
                gamma = config.get('env', {}).get('gamma', 0.99)
                
                if td_lambda_enabled:
                    # 将 dones 和 apply_gamma 转换为 numpy 数组
                    dones_arr = np.array(dones, dtype=np.bool_)
                    apply_gamma_arr = np.array(apply_gamma, dtype=np.bool_)
                    
                    td_returns = compute_td_lambda_returns(
                        kyoku_rewards=kyoku_rewards,
                        at_kyoku=at_kyoku,
                        dones=dones_arr,
                        apply_gamma=apply_gamma_arr,
                        gamma=gamma,
                        lambda_=td_lambda,
                        game_size=game_size
                    )
                else:
                    td_returns = None

                # PERF-03: pre-compute per-step arrays, minimize Python-loop overhead
                actions_np = np.asarray(actions, dtype=np.int64)
                at_kyoku_np = np.asarray(at_kyoku, dtype=np.int64)

                # next_kyoku_idx (clamped)
                next_kyoku_idxs = np.minimum(at_kyoku_np + 1, len(player_ranks) - 1)
                ranks_per_step = player_ranks[next_kyoku_idxs]

                # step returns
                if td_returns is not None:
                    returns_per_step = td_returns
                else:
                    returns_per_step = kyoku_rewards[at_kyoku_np]

                # dq_bonus & dq_best_suit (vectorized)
                is_dq_action = (actions_np == 31) | (actions_np == 32) | (actions_np == 33)
                dq_bonus_arr = np.zeros(game_size, dtype=np.float64)
                dq_best_suit_arr = np.full(game_size, -1, dtype=np.int64)
                if player_dq_quality is not None and is_dq_action.any():
                    dq_mask = is_dq_action & (at_kyoku_np < len(player_dq_quality))
                    if dq_mask.any():
                        dq_bonus_arr[dq_mask] = player_dq_quality[at_kyoku_np[dq_mask]] * ding_que_aux_scale
                if player_dq_best_suit is not None and is_dq_action.any():
                    dq_mask2 = is_dq_action & (at_kyoku_np < len(player_dq_best_suit))
                    if dq_mask2.any():
                        dq_best_suit_arr[dq_mask2] = player_dq_best_suit[at_kyoku_np[dq_mask2]]

                # Target Network: precompute per-step 1-step TD ingredients
                # q_target = imm_reward + bootstrap_discount * V_target(s') + dq_bonus
                target_network_enabled = config.get('target_network', {}).get('enabled', False)
                if target_network_enabled:
                    obs_shape_tuple = obs[0].shape
                    mask_len = len(masks[0])
                    next_obs_arr = []
                    next_masks_arr = []
                    bootstrap_discount_arr = np.zeros(game_size, dtype=np.float64)
                    imm_reward_arr = np.zeros(game_size, dtype=np.float64)
                    dones_list = list(dones) if isinstance(dones, bytes) else dones
                    ag_list = list(apply_gamma) if isinstance(apply_gamma, bytes) else apply_gamma
                    for i in range(game_size):
                        is_done = bool(dones_list[i])
                        if is_done:
                            imm_reward_arr[i] = kyoku_rewards[at_kyoku_np[i]]
                            bootstrap_discount_arr[i] = 0.0
                        else:
                            imm_reward_arr[i] = 0.0
                            bootstrap_discount_arr[i] = gamma if bool(ag_list[i]) else 1.0
                        if i + 1 < game_size and not is_done:
                            next_obs_arr.append(obs[i + 1])
                            next_masks_arr.append(masks[i + 1])
                        else:
                            next_obs_arr.append(np.zeros(obs_shape_tuple, dtype=np.float32))
                            next_masks_arr.append(np.zeros(mask_len, dtype=np.bool_))

                # Oracle Guiding: store invisible_obs for distillation
                oracle_guiding_enabled = config.get('oracle', {}).get('enabled', False)

                # Build buffer entries
                buf = self.buffer
                for i in range(game_size):
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        returns_per_step[i],
                        ranks_per_step[i],
                        dq_bonus_arr[i],
                        dq_best_suit_arr[i],
                        np.asarray(opponent_waits[i], dtype=np.float32),
                    ]
                    if target_network_enabled:
                        entry.extend([
                            next_obs_arr[i],
                            next_masks_arr[i],
                            bootstrap_discount_arr[i],
                            imm_reward_arr[i],
                        ])
                    if oracle_guiding_enabled and self.oracle:
                        entry.append(invisible_obs[i])
                    buf.append(entry)


    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    # Ensure libblood module is available in worker processes
    # This is necessary when using 'spawn' multiprocessing method
    # Import prelude to ensure libblood is properly initialized
    try:
        import prelude  # This will initialize libblood
    except ImportError:
        # Fallback: try to import libblood directly
        import sys
        import os
        try:
            import libblood_loader
        except ImportError:
            pass
        try:
            import libblood
        except ImportError:
            # If libblood is not found, try to add the project root to path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            import libblood
    
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
