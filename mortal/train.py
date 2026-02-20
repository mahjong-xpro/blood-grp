def train():
    import prelude

    import copy
    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    import torch.nn.functional as F
    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer, TrainPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from libblood.consts import obs_shape
    from config import config

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    ding_que_ce_weight = config['aux'].get('ding_que_ce_weight', 0.0)
    ding_que_dqn_ce_weight = config['aux'].get('ding_que_dqn_ce_weight', 0.0)
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = [float(p) for p in config['env']['pts']]
    if len(pts) != 4:
        raise ValueError(f"env.pts must contain exactly 4 values, got {config['env']['pts']}")
    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    # AuxNet dims: (4 = next_rank, 81 = opponent waits: 3 opponents × 27 tiles, 3 = ding_que: Man/Pin/Sou)
    # MODEL-03 fix: 定缺分类头从 DQN Q 值移到 AuxNet 独立分支，避免 CE 与 Bellman 冲突
    aux_net = AuxNet((4, 81, 3)).to(device)

    # Oracle Guiding: perfect-information teacher model
    oracle_config = config.get('oracle', {})
    oracle_enabled = oracle_config.get('enabled', False)
    oracle_distill_weight = oracle_config.get('distill_weight', 1.0)
    oracle_distill_temp = oracle_config.get('distill_temperature', 1.0)
    oracle_dqn_weight = oracle_config.get('oracle_dqn_weight', 1.0)
    oracle_mortal = None
    oracle_dqn = None
    if oracle_enabled:
        oracle_mortal = Brain(
            version=version,
            conv_channels=oracle_config.get('conv_channels', config['resnet']['conv_channels']),
            num_blocks=oracle_config.get('num_blocks', 15),
            is_oracle=True,
        ).to(device)
        oracle_dqn = DQN(version=version).to(device)

    student_models = (mortal, dqn, aux_net)
    oracle_models = (oracle_mortal, oracle_dqn) if oracle_enabled else ()
    all_models = student_models + oracle_models
    if enable_compile:
        for m in all_models:
            m.compile()

    # Target Network: delayed copy of Brain+DQN for stable Q-target computation
    tn_config = config.get('target_network', {})
    target_network_enabled = tn_config.get('enabled', False)
    target_tau = tn_config.get('tau', 0.005)
    target_update_every = tn_config.get('update_every', 1)
    target_mortal = None
    target_dqn = None
    if target_network_enabled:
        target_mortal = copy.deepcopy(mortal).to(device)
        target_dqn = copy.deepcopy(dqn).to(device)
        target_mortal.eval()
        target_dqn.eval()
        for p in target_mortal.parameters():
            p.requires_grad_(False)
        for p in target_dqn.parameters():
            p.requires_grad_(False)

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')
    if target_network_enabled:
        logging.info(f'target network: enabled (tau={target_tau}, update_every={target_update_every})')
    if oracle_enabled:
        logging.info(f'oracle mortal params: {parameter_count(oracle_mortal):,}')
        logging.info(f'oracle dqn params: {parameter_count(oracle_dqn):,}')
        logging.info(f'oracle: enabled (distill_weight={oracle_distill_weight}, temp={oracle_distill_temp}, dqn_weight={oracle_dqn_weight})')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    def build_param_groups(models):
        decay, no_decay = [], []
        for model in models:
            params_dict = {}
            to_decay = set()
            for mod_name, mod in model.named_modules():
                for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                    params_dict[name] = param
                    if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                        to_decay.add(name)
            decay.extend(params_dict[name] for name in sorted(to_decay))
            no_decay.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay},
        ]

    optimizer = optim.AdamW(build_param_groups(student_models), lr=1, weight_decay=0, betas=betas, eps=eps)
    oracle_optimizer = None
    if oracle_enabled:
        oracle_optimizer = optim.AdamW(build_param_groups(oracle_models), lr=1, weight_decay=0, betas=betas, eps=eps)
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        try:
            optimizer.load_state_dict(state['optimizer'])
        except (ValueError, RuntimeError) as e:
            logging.warning(f'Optimizer state incompatible (likely model structure change): {e}')
            logging.warning('Reinitializing optimizer — Adam momentum will restart from scratch')
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']
        if target_network_enabled:
            if 'target_mortal' in state and 'target_dqn' in state:
                target_mortal.load_state_dict(state['target_mortal'])
                target_dqn.load_state_dict(state['target_dqn'])
                logging.info('target network: loaded from checkpoint')
            else:
                target_mortal.load_state_dict(state['mortal'])
                target_dqn.load_state_dict(state['current_dqn'])
                logging.info('target network: initialized from online network (no prior target state)')
        if oracle_enabled:
            if 'oracle_mortal' in state and 'oracle_dqn' in state:
                try:
                    oracle_mortal.load_state_dict(state['oracle_mortal'])
                    oracle_dqn.load_state_dict(state['oracle_dqn'])
                    logging.info('oracle: loaded from checkpoint')
                except RuntimeError as e:
                    logging.warning(f'oracle: architecture changed, reinitializing from scratch: {e}')
            else:
                logging.info('oracle: initialized from scratch (no prior oracle state in checkpoint)')
            if 'oracle_optimizer' in state and oracle_optimizer is not None:
                try:
                    oracle_optimizer.load_state_dict(state['oracle_optimizer'])
                    logging.info('oracle optimizer: loaded from checkpoint')
                except (ValueError, RuntimeError) as e:
                    logging.warning(f'oracle optimizer: incompatible, reinitializing: {e}')
            else:
                logging.info('oracle optimizer: initialized from scratch')
    
    # Scheduler 重启设计（TRAIN-04 文档化）:
    # 从 checkpoint 恢复时，scheduler **不**从 state_dict 恢复，而是用当前
    # config['optim']['scheduler'] 参数 + offset=steps 重新创建。
    #
    # 这是有意设计，支持"阶段切换"（Phase switching）：
    #   - 修改 config.toml 中的 peak/final/warm_up_steps/max_steps 后重启训练，
    #     新 LR 曲线立即生效，从 step=0 开始 warmup（offset 仅用于内部步数计算）。
    #   - 例如 Phase 7C: 将 peak 从 5e-4 降为 2e-4，max_steps 延长到 1.35M。
    #
    # 注意：如果不修改 scheduler config 就重启训练，LR 曲线与中断前完全一致
    # （因为 offset=steps 使 _step_inner 计算出相同的 LR 值）。
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, offset=steps, **config['optim']['scheduler'])
    oracle_scheduler = None
    if oracle_enabled and oracle_optimizer is not None:
        oracle_scheduler = LinearWarmUpCosineAnnealingLR(oracle_optimizer, offset=0, **config['optim']['scheduler'])

    optimizer.zero_grad(set_to_none=True)
    if oracle_optimizer is not None:
        oracle_optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
        'ding_que_ce_loss': 0,
        'opp_wait_loss': 0,
        'ding_que_dqn_ce_loss': 0,
        'ding_que_aux_match': 0,
        'ding_que_dqn_match': 0,
        'ding_que_total': 0,
        'oracle_dqn_loss': 0,
        'distill_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

        # BUG-01 fix: flag to signal epoch restart after test_play.
        # test_play 运行 Rust arena (pyo3 + rayon) 后，DataLoader worker 进程
        # 可能因 GIL 竞争或共享内存管道损坏而无法恢复迭代。
        # 设置此标志后 train_batch() 提前 return → 外层 for 循环 break
        # → train_epoch() 返回 → while True 重新调用 train_epoch() 创建全新 DataLoader。
        _need_restart = False

        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        # 如果文件列表为空（首次训练），自动生成初始数据
        if len(file_list) == 0 and not online:
            logging.warning('No training data found. Generating initial data through self-play...')
            train_player = TrainPlayer()
            rankings, generated_files = train_player.train_play(mortal, dqn, device)
            logging.info(f'Generated {len(generated_files)} files from self-play')
            # BUG-08 fix: rankings=[1st_count,2nd_count,...], .mean() 是各档平均局数, 非平均排名
            avg_rank = np.dot(rankings, [1, 2, 3, 4]) / max(rankings.sum(), 1)
            logging.info(f'Average ranking: {avg_rank:.4f} (distribution: {rankings})')
            
            # 重新构建文件索引
            logging.info('Rebuilding file index with generated data...')
            file_list = []
            for pat in config['dataset']['globs']:
                file_list.extend(glob(pat, recursive=True))
            # 如果之前有 player_names_set，需要重新过滤
            if 'player_names_set' in locals() and len(player_names_set) > 0:
                filtered = []
                for filename in tqdm(file_list, unit='file'):
                    with gzip.open(filename, 'rt') as f:
                        start = json.loads(next(f))
                        if not set(start['names']).isdisjoint(player_names_set):
                            filtered.append(filename)
                file_list = filtered
            file_list.sort(reverse=True)
            torch.save({'file_list': file_list}, file_index)
            logging.info(f'File list size after self-play: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            oracle = True,
            trust_seed = True,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
        ))

        remaining_batches_list = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        # Number of fields per buffer entry: base(9) + target_network(0|4) + oracle(0|1)
        _n_base_fields = 9
        _n_tn_fields = 4 if target_network_enabled else 0
        _n_oracle_fields = 1 if oracle_enabled else 0

        def train_batch(obs, actions, masks, steps_to_done, returns, player_ranks,
                        ding_que_bonus, ding_que_best_suit, opponent_waits,
                        next_obs=None, next_masks=None, bootstrap_discount=None, imm_reward=None,
                        invisible_obs=None):
            nonlocal steps
            nonlocal idx
            nonlocal pb
            nonlocal _need_restart

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            returns = returns.to(dtype=torch.float64, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            ding_que_bonus = ding_que_bonus.to(dtype=torch.float32, device=device)
            ding_que_best_suit = ding_que_best_suit.to(dtype=torch.int64, device=device)
            opponent_waits = opponent_waits.to(dtype=torch.float32, device=device)

            valid = masks[range(batch_size), actions]
            if not valid.all():
                invalid_count = (~valid).sum().item()
                first_invalid = (~valid).nonzero(as_tuple=True)[0][0].item()
                msg = (f"Skipping batch at step {steps + 1}: {invalid_count}/{batch_size} samples have action not allowed by mask "
                       f"(first invalid idx={first_invalid}, action={actions[first_invalid].item()}). "
                       f"Likely bad log or ding_que replay bug.")
                logging.error(msg)
                raise RuntimeError(msg)

            # Q-target 计算:
            # - Target Network 启用时: td_lambda 混合 MC 回报与 1-step TD
            #     td1 = r_imm + discount * V_target(s')
            #     mc  = returns (MC 或 TD(λ) 回报)
            #     q_target = λ * mc + (1-λ) * td1 + dq_bonus
            #     λ=1.0 → 纯 MC (等同于未启用 TN), λ=0.0 → 纯 1-step TD
            # - 否则: MC 回报 (向后兼容)
            if target_network_enabled and next_obs is not None:
                next_obs_t = next_obs.to(dtype=torch.float32, device=device)
                next_masks_t = next_masks.to(dtype=torch.bool, device=device)
                bd = bootstrap_discount.to(dtype=torch.float32, device=device)
                ir = imm_reward.to(dtype=torch.float32, device=device)
                with torch.no_grad():
                    phi_next = target_mortal(next_obs_t)
                    q_next_all = target_dqn(phi_next, next_masks_t)
                    has_valid = next_masks_t.any(dim=-1)
                    v_next = torch.where(
                        has_valid,
                        q_next_all.masked_fill(~next_masks_t, -torch.inf).max(dim=-1).values,
                        torch.zeros_like(bd),
                    )
                td1_target = ir + bd * v_next
                # BUG-5 fix: dataloader 现在始终提供原始 kyoku 奖励（TN 启用时），
                # 此处统一应用 γ^n 折扣，λ 仅作为 MC/TD1 混合权重（不再双重衰减）
                mc_target = (gamma ** steps_to_done * returns).to(dtype=torch.float32, device=device)
                td_lambda_val = config.get('env', {}).get('td_lambda', 1.0)
                q_target = td_lambda_val * mc_target + (1.0 - td_lambda_val) * td1_target + ding_que_bonus
            else:
                td_lambda_enabled = config.get('env', {}).get('td_lambda_enabled', False)
                if td_lambda_enabled:
                    q_target = returns + ding_que_bonus
                else:
                    q_target = gamma ** steps_to_done * returns + ding_que_bonus
            q_target = q_target.to(torch.float32)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                dqn_loss = 0.5 * mse(q, q_target)
                cql_loss = 0
                if not online:
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                next_rank_logits, opp_wait_logits, ding_que_logits = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)
                
                # Opponent wait prediction BCE loss
                opp_wait_enabled = config.get('aux', {}).get('opp_wait_enabled', False)
                opp_wait_weight = config.get('aux', {}).get('opp_wait_weight', 0.1)
                # FIX: 移除多余的 'opponent_waits' in locals() 检查。
                # opponent_waits 是 train_batch() 的参数，始终存在于 locals() 中。
                if opp_wait_enabled:
                    opp_wait_loss = F.binary_cross_entropy_with_logits(
                        opp_wait_logits, opponent_waits.float()
                    )
                else:
                    opp_wait_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                # MODEL-03 fix: 定缺 CE 主要作用于 AuxNet 独立分类头的 logits，
                # 避免 CE 与 Bellman 目标的梯度冲突。
                # 同时对 DQN 的定缺 Q 值施加弱 CE，给予方向信号。
                sel = ding_que_best_suit >= 0
                if sel.any() and ding_que_ce_weight > 0:
                    ding_que_ce_loss = ce(ding_que_logits[sel], ding_que_best_suit[sel])
                else:
                    ding_que_ce_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                # DQN 定缺弱 CE: 对 DQN 在 action 31/32/33 的 Q 值施加轻量级 CE，
                # 让 DQN 直接获得定缺方向信号，而非仅依赖微弱的 ding_que_bonus (±0.02)。
                if sel.any() and ding_que_dqn_ce_weight > 0:
                    dqn_dq_logits = q_out[sel][:, 31:34]  # DQN 在万/筒/条定缺动作上的 Q 值
                    ding_que_dqn_ce_loss = ce(dqn_dq_logits, ding_que_best_suit[sel])
                else:
                    ding_que_dqn_ce_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                # 定缺匹配率: 分别跟踪 AuxNet 和 DQN 的准确率
                if sel.any():
                    # AuxNet 分类准确率 (跟踪 CE loss 效果)
                    aux_pred = ding_que_logits[sel].argmax(-1)
                    stats['ding_que_aux_match'] += (aux_pred == ding_que_best_suit[sel]).sum().item()
                    # DQN 实际决策准确率 (跟踪 gameplay 效果)
                    dqn_suit = actions[sel] - 31
                    stats['ding_que_dqn_match'] += (dqn_suit == ding_que_best_suit[sel]).sum().item()
                    stats['ding_que_total'] += sel.sum().item()

                # Oracle Guiding: teacher forward pass + distillation
                if oracle_enabled and invisible_obs is not None:
                    inv_obs = invisible_obs.to(dtype=torch.float32, device=device)
                    oracle_phi = oracle_mortal(obs, inv_obs)
                    oracle_q_out = oracle_dqn(oracle_phi, masks)
                    oracle_q = oracle_q_out[range(batch_size), actions]
                    o_dqn_loss = 0.5 * mse(oracle_q, q_target)

                    # ISSUE-1 fix: 定缺步排除蒸馏。定缺步仅允许 action 31/32/33，
                    # Oracle 无显式定缺 CE 监督，其定缺 Q 值不可靠，
                    # 蒸馏会破坏 Student 已学好的定缺策略。
                    is_ding_que_only = (masks[:, :31].sum(-1) == 0) & (masks[:, 34:].sum(-1) == 0)
                    distill_select = ~is_ding_que_only

                    if distill_select.any():
                        # KL distillation: replace -inf with -1e9 to avoid NaN.
                        d_masks = masks[distill_select]
                        oracle_q_safe = oracle_q_out[distill_select].masked_fill(~d_masks, -1e9)
                        student_q_safe = q_out[distill_select].masked_fill(~d_masks, -1e9)
                        with torch.no_grad():
                            oracle_policy = F.softmax(oracle_q_safe / oracle_distill_temp, dim=-1)
                        student_log_policy = F.log_softmax(student_q_safe / oracle_distill_temp, dim=-1)
                        distill_loss = F.kl_div(student_log_policy, oracle_policy, reduction='batchmean')
                    else:
                        distill_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                else:
                    o_dqn_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                    distill_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                student_loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                    next_rank_loss * next_rank_weight,
                    ding_que_ce_loss * ding_que_ce_weight,
                    ding_que_dqn_ce_loss * ding_que_dqn_ce_weight,
                    opp_wait_loss * opp_wait_weight,
                    distill_loss * oracle_distill_weight if oracle_enabled else 0,
                ))
                oracle_loss = o_dqn_loss * oracle_dqn_weight if oracle_enabled else torch.tensor(0.0)
                loss = student_loss + oracle_loss
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                if not online:
                    stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                stats['ding_que_ce_loss'] += ding_que_ce_loss
                stats['ding_que_dqn_ce_loss'] += ding_que_dqn_ce_loss
                stats['opp_wait_loss'] += opp_wait_loss
                if oracle_enabled:
                    stats['oracle_dqn_loss'] += o_dqn_loss
                    stats['distill_loss'] += distill_loss
                all_q[idx] = q
                all_q_target[idx] = q_target

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    student_params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(student_params, max_grad_norm)
                    if oracle_optimizer is not None:
                        scaler.unscale_(oracle_optimizer)
                        oracle_params = chain.from_iterable(g['params'] for g in oracle_optimizer.param_groups)
                        clip_grad_norm_(oracle_params, max_grad_norm)
                scaler.step(optimizer)
                if oracle_optimizer is not None:
                    scaler.step(oracle_optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if oracle_optimizer is not None:
                    oracle_optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if oracle_scheduler is not None:
                    oracle_scheduler.step()
                # Target Network 软更新: θ_target ← τ*θ + (1-τ)*θ_target
                if target_network_enabled and steps % target_update_every == 0:
                    with torch.no_grad():
                        for tp, sp in zip(target_mortal.parameters(), mortal.parameters()):
                            tp.data.mul_(1 - target_tau).add_(sp.data, alpha=target_tau)
                        for tp, sp in zip(target_dqn.parameters(), dqn.parameters()):
                            tp.data.mul_(1 - target_tau).add_(sp.data, alpha=target_tau)
                        for tb, sb in zip(target_mortal.buffers(), mortal.buffers()):
                            tb.data.copy_(sb.data)
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # downsample to reduce tensorboard event size
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                if not online:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                writer.add_scalar('loss/ding_que_ce_loss', stats['ding_que_ce_loss'] / save_every, steps)
                writer.add_scalar('loss/ding_que_dqn_ce_loss', stats['ding_que_dqn_ce_loss'] / save_every, steps)
                writer.add_scalar('loss/opp_wait_loss', stats['opp_wait_loss'] / save_every, steps)
                if oracle_enabled:
                    writer.add_scalar('loss/oracle_dqn_loss', stats['oracle_dqn_loss'] / save_every, steps)
                    writer.add_scalar('loss/distill_loss', stats['distill_loss'] / save_every, steps)
                # 定缺匹配率: aux = AuxNet 分类头准确率, dqn = DQN 实际动作准确率
                dq_total = stats['ding_que_total']
                if dq_total > 0:
                    writer.add_scalar('ding_que/aux_match_rate', stats['ding_que_aux_match'] / dq_total, steps)
                    writer.add_scalar('ding_que/dqn_match_rate', stats['ding_que_dqn_match'] / dq_total, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                }
                if target_network_enabled:
                    state['target_mortal'] = target_mortal.state_dict()
                    state['target_dqn'] = target_dqn.state_dict()
                if oracle_enabled:
                    state['oracle_mortal'] = oracle_mortal.state_dict()
                    state['oracle_dqn'] = oracle_dqn.state_dict()
                    if oracle_optimizer is not None:
                        state['oracle_optimizer'] = oracle_optimizer.state_dict()
                torch.save(state, state_file)
                
                # 每 test_every 步保存一个历史 checkpoint，与 test_play 同步
                if steps % test_every == 0:
                    checkpoint_dir = '/data/mortal/checkpoints'
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_file = f'{checkpoint_dir}/mortal_{steps // 1000}k.pth'
                    torch.save(state, checkpoint_file)
                    logging.info(f'Checkpoint saved: {checkpoint_file}')

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, dqn, device)
                    mortal.train()
                    dqn.train()

                    avg_pt = stat.avg_pt(pts)
                    # FIX: 原 AND 条件要求 avg_pt 和 avg_rank 同时改善（Pareto improvement），
                    # 实际中两个指标存在评估噪声，同时改善极难触发，导致 best 模型长期不更新。
                    # 改为以 avg_rank 为主指标（越低越好），avg_pt 为辅助参考。
                    better = stat.avg_rank < best_perf['avg_rank'] or \
                             (stat.avg_rank == best_perf['avg_rank'] and avg_pt > best_perf['avg_pt'])
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                    }, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    
                    # 自动更新 baseline (阶梯式训练)
                    # 当 avg_pt 超过阈值时，用当前模型替换 baseline，创造持续进步压力
                    baseline_config = config.get('baseline', {}).get('train', {})
                    baseline_file = baseline_config.get('state_file', '/data/mortal/baseline.pth')
                    auto_update_threshold = config.get('reward_shaping', {}).get('baseline_update_threshold', 3.2)
                    if avg_pt >= auto_update_threshold and better:
                        shutil.copy(state_file, baseline_file)
                        logging.info(f'Baseline updated: avg_pt={avg_pt:.4} >= threshold={auto_update_threshold}')
                        writer.add_scalar('baseline/update_step', steps, steps)
                        # FIX: 自动更新 baseline 文件后，必须刷新内存中的 baseline engine。
                        # 之前只写文件不重新加载，导致离线模式的阶梯式训练完全失效
                        # （所有后续自对弈仍使用旧 baseline 权重）。
                        if train_player is not None:
                            train_player.reload_baseline(baseline_file)
                        # FIX BUG-01: 进程不再重启，test_player 的 baseline 也需手动刷新，
                        # 确保下次 test_play 使用最新 baseline 模型。
                        test_player.reload_baseline(baseline_file)
                    
                    if online:
                        # FIX BUG-01: 原 workaround 用 sys.exit(0) 杀进程再由父进程重启。
                        # 根因：test_play 运行 Rust arena (pyo3 + rayon) 后，DataLoader
                        # worker 进程可能因 GIL 竞争或共享内存管道损坏而无法恢复迭代。
                        # 修复：设置标志 → 外层 for 循环 break → train_epoch() 返回
                        # → while True 循环重新调用 train_epoch() 创建全新 DataLoader。
                        logging.info('Online: recycling DataLoader after test_play')
                        _need_restart = True
                        return
                pb = tqdm(total=save_every, desc='TRAIN')

        def _unpack_batch(batch_tuple):
            idx = _n_base_fields
            base = batch_tuple[:idx]
            if target_network_enabled and len(batch_tuple) > idx:
                tn = batch_tuple[idx:idx + _n_tn_fields]
                idx += _n_tn_fields
            else:
                tn = (None, None, None, None)
            if oracle_enabled and len(batch_tuple) > idx:
                oracle = (batch_tuple[idx],)
            else:
                oracle = (None,)
            return (*base, *tn, *oracle)

        for batch_tuple in data_loader:
            bs = batch_tuple[0].shape[0]
            if bs != batch_size:
                remaining_batches_list.append(batch_tuple)
                remaining_bs += bs
                continue
            train_batch(*_unpack_batch(batch_tuple))
            if _need_restart:
                break

        if _need_restart:
            return

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            catted = []
            for fi in range(len(remaining_batches_list[0])):
                catted.append(torch.cat([b[fi] for b in remaining_batches_list], dim=0))
            catted = tuple(catted)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                sliced = tuple(c[start:end] for c in catted)
                train_batch(*_unpack_batch(sliced))
                if _need_restart:
                    break
                start = end
                end += batch_size
        if _need_restart:
            return
        pb.close()

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    # Initialize train_player for offline mode self-play
    train_player = None
    if not online:
        train_player = TrainPlayer()

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        
        # In offline mode, generate new self-play data after each epoch
        if not online:
            logging.info('Epoch completed. Generating new self-play data...')
            rankings, generated_files = train_player.train_play(mortal, dqn, device)
            logging.info(f'Generated {len(generated_files)} files from self-play')
            # BUG-08 fix: 同上
            avg_rank = np.dot(rankings, [1, 2, 3, 4]) / max(rankings.sum(), 1)
            logging.info(f'Average ranking: {avg_rank:.4f} (distribution: {rankings})')
            
            # PERF-04: 增量更新文件索引，仅追加新生成的文件（不再全量 glob + sort）。
            # generated_files 已由 train_play() 返回，包含本次新增的完整路径。
            if generated_files:
                old_size = len(file_list)
                file_list.extend(generated_files)
                file_index = config['dataset']['file_index']
                torch.save({'file_list': file_list}, file_index)
                logging.info(f'File index updated: {old_size:,} + {len(generated_files)} new → {len(file_list):,} total')
            logging.info('Starting next epoch with updated data...')

def main():
    import os
    import sys
    import time
    import multiprocessing
    from subprocess import Popen
    from config import config

    # Set multiprocessing start method to 'spawn' to avoid fork() issues
    # This prevents the deprecation warning about multi-threaded processes using fork()
    # Required on both macOS and Linux when using PyTorch DataLoader with num_workers > 0
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

    # Online 模式通过子进程运行 train()。
    # BUG-01 已修复（不再需要 sys.exit(0) + 重启循环），但保留子进程包装
    # 作为崩溃恢复安全网：如果 train() 异常退出，父进程以相同错误码退出。
    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
