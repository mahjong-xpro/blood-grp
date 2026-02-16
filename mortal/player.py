import torch
import numpy as np
import os
import shutil
import secrets
import logging
import glob
import random
from os import path
from model import Brain, DQN
from engine import MortalEngine
from libblood.stat import Stat
from libblood.arena import OneVsThree
from config import config

class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        device = torch.device(baseline_cfg['device'])
        baseline_file = baseline_cfg['state_file']

        # 如果 baseline 文件不存在，尝试使用当前模型文件
        if not path.exists(baseline_file):
            current_state_file = config['control']['state_file']
            if path.exists(current_state_file):
                logging.warning(f'baseline file not found: {baseline_file}, using current model: {current_state_file}')
                baseline_file = current_state_file
            else:
                # 如果当前模型也不存在，创建随机初始化的模型
                logging.warning(f'baseline file not found: {baseline_file}, creating random baseline model')
                version = config['control']['version']
                conv_channels = config['resnet']['conv_channels']
                num_blocks = config['resnet']['num_blocks']
                stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
                stable_dqn = DQN(version=version).eval()
                if baseline_cfg['enable_compile']:
                    stable_mortal.compile()
                    stable_dqn.compile()
                self.baseline_engine = MortalEngine(
                    stable_mortal,
                    stable_dqn,
                    is_oracle = False,
                    version = version,
                    device = device,
                    enable_amp = False,
                    enable_rule_based_agari_guard = True,
                    name = 'baseline',
                )
                self.chal_version = config['control']['version']
                self.log_dir = path.abspath(config['test_play']['log_dir'])
                self._baseline_cfg = baseline_cfg
                return

        state = torch.load(baseline_file, weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = False,
            enable_rule_based_agari_guard = True,
            name = 'baseline',
        )
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(config['test_play']['log_dir'])

        # 保存 baseline 配置，供 reload_baseline 使用
        self._baseline_cfg = config['baseline']['test']

    def reload_baseline(self, baseline_file=None):
        """重新加载 baseline 模型权重（BUG-01 fix: 进程不再重启，需手动刷新 baseline）。"""
        cfg = self._baseline_cfg
        if baseline_file is None:
            baseline_file = cfg['state_file']
        if not path.exists(baseline_file):
            logging.warning(f'TestPlayer.reload_baseline: file not found: {baseline_file}')
            return
        device = torch.device(cfg['device'])
        state = torch.load(baseline_file, weights_only=True, map_location=torch.device('cpu'))
        model_cfg = state['config']
        version = model_cfg['control'].get('version', 1)
        conv_channels = model_cfg['resnet']['conv_channels']
        num_blocks = model_cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
        if cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()
        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle=False,
            version=version,
            device=device,
            enable_amp=False,
            enable_rule_based_agari_guard=True,
            name='baseline',
        )
        logging.info(f'TestPlayer baseline reloaded from {baseline_file}')

    def test_play(self, seed_count, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        # FIX: 与 baseline 保持一致，启用 rule_based_agari_guard，
        # 否则 challenger 缺少和牌安全网而 baseline 有，评估不公平。
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            device = device,
            enable_amp = False,
            enable_rule_based_agari_guard = True,
            name = 'mortal',
        )

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        # Test play always uses default rules (no randomization) for fair evaluation
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
            randomize_fan_config = False,
        )
        env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (10000, 0x2000),
            seed_count = seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return stat

class TrainPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['train']
        device = torch.device(baseline_cfg['device'])
        baseline_file = baseline_cfg['state_file']

        # 如果 baseline 文件不存在，尝试使用当前模型文件
        if not path.exists(baseline_file):
            current_state_file = config['control']['state_file']
            if path.exists(current_state_file):
                logging.warning(f'baseline file not found: {baseline_file}, using current model: {current_state_file}')
                baseline_file = current_state_file
            else:
                # 如果当前模型也不存在，创建随机初始化的模型
                logging.warning(f'baseline file not found: {baseline_file}, creating random baseline model')
                version = config['control']['version']
                conv_channels = config['resnet']['conv_channels']
                num_blocks = config['resnet']['num_blocks']
                stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
                stable_dqn = DQN(version=version).eval()
                if baseline_cfg['enable_compile']:
                    stable_mortal.compile()
                    stable_dqn.compile()
                self.baseline_engine = MortalEngine(
                    stable_mortal,
                    stable_dqn,
                    is_oracle = False,
                    version = version,
                    device = device,
                    enable_amp = False,
                    enable_rule_based_agari_guard = True,
                    name = 'baseline',
                )
                profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
                logging.info(f'using profile {profile}')
                cfg = config['train_play'][profile]
                self.chal_version = config['control']['version']
                self.log_dir = path.abspath(cfg['log_dir'])
                self.train_key = secrets.randbits(64)
                self.train_seed = 10000
                self.seed_count = cfg['games'] // 4
                self.boltzmann_epsilon = cfg['boltzmann_epsilon']
                self.boltzmann_temp = cfg['boltzmann_temp']
                self.top_p = cfg['top_p']
                self.keep_data = cfg.get('keep_data', False)
                self.repeats = cfg['repeats']
                self.repeat_counter = 0
                # BUG-07 fix: 必须在 return 前设置, 否则 reload_baseline() 会 AttributeError
                self._baseline_cfg = config['baseline']['train']
                return

        state = torch.load(baseline_file, weights_only=True, map_location=torch.device('cpu'))
        cfg = state['config']
        version = cfg['control'].get('version', 1)
        conv_channels = cfg['resnet']['conv_channels']
        num_blocks = cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
        if baseline_cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()

        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = False,
            enable_rule_based_agari_guard = True,
            name = 'baseline',
        )

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(cfg['log_dir'])
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000

        self.seed_count = cfg['games'] // 4
        self.boltzmann_epsilon = cfg['boltzmann_epsilon']
        self.boltzmann_temp = cfg['boltzmann_temp']
        self.top_p = cfg['top_p']
        self.keep_data = cfg.get('keep_data', False)

        self.repeats = cfg['repeats']
        self.repeat_counter = 0

        # 保存 baseline 配置，供 reload_baseline 使用
        self._baseline_cfg = config['baseline']['train']

    def reload_baseline(self, baseline_file=None):
        """重新加载 baseline 模型权重（阶梯式训练：自动更新后需要刷新内存中的 baseline）。"""
        cfg = self._baseline_cfg
        if baseline_file is None:
            baseline_file = cfg['state_file']
        if not path.exists(baseline_file):
            logging.warning(f'reload_baseline: file not found: {baseline_file}')
            return
        device = torch.device(cfg['device'])
        state = torch.load(baseline_file, weights_only=True, map_location=torch.device('cpu'))
        model_cfg = state['config']
        version = model_cfg['control'].get('version', 1)
        conv_channels = model_cfg['resnet']['conv_channels']
        num_blocks = model_cfg['resnet']['num_blocks']
        stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        stable_dqn = DQN(version=version).eval()
        stable_mortal.load_state_dict(state['mortal'])
        stable_dqn.load_state_dict(state['current_dqn'])
        if cfg['enable_compile']:
            stable_mortal.compile()
            stable_dqn.compile()
        self.baseline_engine = MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = False,
            enable_rule_based_agari_guard = True,
            name = 'baseline',
        )
        logging.info(f'Baseline engine reloaded from {baseline_file}')

    def _select_from_pool(self):
        """从对手池中加权随机选择一个检查点文件。
        最新文件（按修改时间）权重 = newest_weight，其他文件权重 = 1.0。
        如果池为空或不存在，返回默认 baseline 路径。
        """
        pool_cfg = config.get('baseline', {}).get('pool', {})
        pool_dir = pool_cfg.get('pool_dir', '')
        newest_weight = pool_cfg.get('newest_weight', 3.0)

        if not pool_dir or not path.isdir(pool_dir):
            return self._baseline_cfg['state_file']

        files = glob.glob(path.join(pool_dir, '*.pth'))
        if not files:
            logging.warning(f'Baseline pool is empty: {pool_dir}, using default baseline')
            return self._baseline_cfg['state_file']

        # 按修改时间排序（最旧在前，最新在后），避免文件名字典序排序错误
        # 例如 mortal_80k.pth 字典序 > mortal_235k.pth，但实际是更旧的
        files.sort(key=lambda f: os.path.getmtime(f))

        weights = [1.0] * len(files)
        weights[-1] = newest_weight
        chosen = random.choices(files, weights=weights, k=1)[0]
        logging.info(f'Selected baseline from pool: {path.basename(chosen)} (pool size={len(files)})')
        return chosen

    def reload_baseline_from_pool(self):
        """从对手池中随机选取一个检查点并重载为 baseline。"""
        chosen_file = self._select_from_pool()
        self.reload_baseline(chosen_file)

    def train_play(self, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            boltzmann_epsilon = self.boltzmann_epsilon,
            boltzmann_temp = self.boltzmann_temp,
            top_p = self.top_p,
            device = device,
            enable_amp = False,
            name = 'trainee',
        )

        if path.isdir(self.log_dir) and not self.keep_data:
            shutil.rmtree(self.log_dir)

        # FIX: keep_data=True 时记录已有文件，避免重复提交旧日志。
        existing_files = set(os.listdir(self.log_dir)) if path.isdir(self.log_dir) else set()

        # Phase 2: multi-rule training — randomize FanConfig per game if configured
        randomize_rules = config.get('rules', {}).get('randomize_fan_config', False)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
            randomize_fan_config = randomize_rules,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        # FIX: 仅返回本次新增的日志文件，排除旧文件和子目录。
        all_entries = os.listdir(self.log_dir) if path.isdir(self.log_dir) else []
        file_list = [
            path.join(self.log_dir, p) for p in all_entries
            if p not in existing_files and path.isfile(path.join(self.log_dir, p))
        ]

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
