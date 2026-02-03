import random
import torch
import numpy as np
import logging
from torch.utils.data import IterableDataset

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
        self.reward_calc = RewardCalculator()

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
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

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
                if len(ding_que_best_suit_list) > 0:
                    dq_best_suit_arr = np.array(ding_que_best_suit_list, dtype=np.int64)
                    player_dq_best_suit = dq_best_suit_arr[:, player_id]
                else:
                    player_dq_best_suit = None

                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                scores_seq = np.concatenate((scores_history, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                # steps_to_done[i] depends on i+1, so compute with a running accumulator
                # to avoid out-of-bounds when i == game_size - 1.
                steps = 0
                for i in reversed(range(game_size)):
                    if dones[i]:
                        steps = 0
                    else:
                        steps += int(apply_gamma[i])
                    steps_to_done[i] = steps

                for i in range(game_size):
                    # player_ranks is based on scores_history + final_scores, so it usually has
                    # length = (#kyoku + 1). Some logs may mark actions with at_kyoku==#kyoku
                    # (final/terminal), so clamp to the last valid index.
                    next_kyoku_idx = int(at_kyoku[i]) + 1
                    if next_kyoku_idx >= len(player_ranks):
                        next_kyoku_idx = len(player_ranks) - 1
                    # Ding Que bonus: only at step where action is DingQue (31=Man, 32=Pin, 33=Sou)
                    if player_dq_quality is not None and int(actions[i]) in (31, 32, 33):
                        k = int(at_kyoku[i])
                        if k < len(player_dq_quality):
                            dq_bonus = float(player_dq_quality[k] * ding_que_aux_scale)
                        else:
                            dq_bonus = 0.0
                    else:
                        dq_bonus = 0.0
                    # Ding Que best-suit label for CE auxiliary: 0=Man, 1=Pin, 2=Sou; -1 = not a DingQue step
                    if player_dq_best_suit is not None and int(actions[i]) in (31, 32, 33):
                        k = int(at_kyoku[i])
                        if k < len(player_dq_best_suit):
                            dq_best_suit = int(player_dq_best_suit[k])
                        else:
                            dq_best_suit = -1
                    else:
                        dq_best_suit = -1
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[next_kyoku_idx],
                        dq_bonus,
                        dq_best_suit,
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

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
