def train():
    import prelude

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
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
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
    aux_net = AuxNet((4,)).to(device)
    
    # Check for multiple GPUs
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training!")
        mortal = nn.DataParallel(mortal)
        dqn = nn.DataParallel(dqn)
        aux_net = nn.DataParallel(aux_net)
        
    all_models = (mortal, dqn, aux_net)
    if enable_compile:
        # Compile underlying modules if DataParallel is used, or the model itself
        for m in all_models:
            if isinstance(m, nn.DataParallel):
                m.module.compile()
            else:
                m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    
    # Access underlying module for param counting if DataParallel
    def get_module(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    logging.info(f'mortal params: {parameter_count(get_module(mortal)):,}')
    logging.info(f'dqn params: {parameter_count(get_module(dqn)):,}')
    logging.info(f'aux params: {parameter_count(get_module(aux_net)):,}')

    get_module(mortal).freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        # Use underlying module for parameter iteration
        real_model = get_module(model)
        for mod_name, mod in real_model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        # Verify keys exist before access (DataParallel safe)
        decay_params.extend(params_dict[name] for name in sorted(to_decay) if name in params_dict)
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay) if name in params_dict)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
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
        
        # Helper to load state dict handling DataParallel prefix
        def load_sd(model, sd):
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(sd)
            else:
                model.load_state_dict(sd)

        load_sd(mortal, state['mortal'])
        load_sd(dqn, state['current_dqn'])
        load_sd(aux_net, state['aux_net'])
        
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        # Submit underlying models
        submit_param(get_module(mortal), get_module(dqn), is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

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
            logging.info(f'Average ranking: {rankings.mean():.2f}')
            
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
            pin_memory = True,  # Speed up Host -> Device transfer
            prefetch_factor = 2 if num_workers > 0 else None,  # Pre-load batches
            persistent_workers = True if num_workers > 0 else False,  # Avoid respawn overhead
            worker_init_fn = worker_init_fn,
        ))

        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)

            # Skip batch if any (state, action) pair is invalid (action not allowed by mask).
            # This can happen when logs contain moves from a buggy version (e.g. discard before
            # ding_que) or when replay state diverges from the state when the action was recorded.
            valid = masks[range(batch_size), actions]
            if not valid.all():
                invalid_count = (~valid).sum().item()
                first_invalid = (~valid).nonzero(as_tuple=True)[0][0].item()
                # Raise RuntimeError to force debugging
                msg = (f"Skipping batch at step {steps + 1}: {invalid_count}/{batch_size} samples have action not allowed by mask "
                       f"(first invalid idx={first_invalid}, action={actions[first_invalid].item()}). "
                       f"Likely bad log or ding_que replay bug.")
                logging.error(msg)
                raise RuntimeError(msg)

            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                dqn_loss = 0.5 * mse(q, q_target_mc)
                cql_loss = 0
                if not online:
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                next_rank_logits, = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)

                loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                    next_rank_loss * next_rank_weight,
                ))
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                if not online:
                    stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(get_module(mortal), get_module(dqn), is_idle=False)
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
                    'mortal': get_module(mortal).state_dict(),
                    'current_dqn': get_module(dqn).state_dict(),
                    'aux_net': get_module(aux_net).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                }
                torch.save(state, state_file)

                if online and steps % submit_every != 0:
                    submit_param(get_module(mortal), get_module(dqn), is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, dqn, device)
                    mortal.train()
                    dqn.train()

                    # Use linear weights for Bloody Battle: 1st=6, 2nd=4, 3rd=2, 4th=0
                    # Average is 3.0. If > 3.0, model is better than random.
                    avg_pt = stat.avg_pt([6, 4, 2, 0])
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
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
                        # Update best.pth
                        torch.save(state, state_file)
                        baseline_file = config['baseline']['train']['state_file']
                        
                        logging.info(
                            'a new record has been made! '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}. '
                            f'Updating {best_state_file} AND {baseline_file} (Evolution Step)'
                        )
                        shutil.copy(state_file, best_state_file)
                        shutil.copy(state_file, baseline_file)
                        
                        # CRITICAL FIX: Reset best_perf after updating baseline!
                        # The opponent just got harder (it's us!). We don't need to beat the OLD score (e.g. 4.5 vs Random),
                        # we just need to beat the NEW baseline (score > 3.0).
                        best_perf = {
                            'avg_rank': 2.5,
                            'avg_pt': 3.0,
                        }
                        logging.info(f"Baseline Updated. Resetting target score to {best_perf['avg_pt']} for next round.")
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks in data_loader:
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                )
                start = end
                end += batch_size
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
            logging.info(f'Average ranking: {rankings.mean():.2f}')
            
            # Rebuild file index with newly generated data
            logging.info('Rebuilding file index with newly generated data...')
            file_index = config['dataset']['file_index']
            file_list = []
            for pat in config['dataset']['globs']:
                file_list.extend(glob(pat, recursive=True))
            
            # Apply player_names filter if needed
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            if len(player_names_set) > 0:
                filtered = []
                for filename in tqdm(file_list, unit='file', desc='Filtering files'):
                    with gzip.open(filename, 'rt') as f:
                        start = json.loads(next(f))
                        if not set(start['names']).isdisjoint(player_names_set):
                            filtered.append(filename)
                file_list = filtered
            
            file_list.sort(reverse=True)
            torch.save({'file_list': file_list}, file_index)
            logging.info(f'File list size after self-play: {len(file_list):,}')
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
    # Fix Multiprocessing Deadlock with CUDA/Rust: Use 'spawn' instead of 'fork'
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    try:
        main()
    except KeyboardInterrupt:
        pass
