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
        # Partition file list: Binary Chunks vs Legacy JSON
        binary_files = [f for f in file_list if f.endswith('.bin.lz4')]
        legacy_files = [f for f in file_list if not f.endswith('.bin.lz4')]

        # 1. Fast Path: Binary Chunks
        for bin_file in binary_files:
            try:
                # BinaryLoader logic moved to GameplayLoader instance method
                games = self.loader.load_binary_chunk(bin_file)
                # Process games (extract training samples)
                for game in games:
                    samples = game.take_batch()
                    if self.oracle:
                         invisible_obs = game.take_invisible_obs()
                         if len(samples) == len(invisible_obs):
                             new_samples = []
                             for i, sample in enumerate(samples):
                                 new_sample = (
                                     sample[0],
                                     invisible_obs[i],
                                     sample[1],
                                     sample[2],
                                     sample[3],
                                     sample[4],
                                     sample[5]
                                 )
                                 new_samples.append(new_sample)
                             samples = new_samples
                    self.buffer.extend(samples)
            except Exception as e:
                logging.warning(f"Failed to load binary chunk {bin_file}: {e}")

        # 2. Legacy Path: JSON GZ Files
        if legacy_files:
            try:
                data = self.loader.load_gz_log_files(legacy_files)
                for file in data:
                    for game in file:
                         samples = game.take_batch()
                         if self.oracle:
                             invisible_obs = game.take_invisible_obs()
                             if len(samples) == len(invisible_obs):
                                 new_samples = []
                                 for i, sample in enumerate(samples):
                                     new_sample = (
                                         sample[0],
                                         invisible_obs[i],
                                         sample[1],
                                         sample[2],
                                         sample[3],
                                         sample[4],
                                         sample[5]
                                     )
                                     new_samples.append(new_sample)
                                 samples = new_samples
                         self.buffer.extend(samples)

            except Exception as e:
                logging.warning(f"Failed to load legacy logs: {e}")
                return
        


    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    # Prevent Rayon Thread Oversubscription
    # Default Rayon launches 128 threads per worker. 
    # 16 Workers * 128 Threads = 2048 Threads -> CPU Thrashing.
    # Limit to 4 threads per worker (16 * 4 = 64 threads total).
    import os
    os.environ['RAYON_NUM_THREADS'] = '4'

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
