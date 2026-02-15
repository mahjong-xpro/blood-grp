#!/usr/bin/env python3
"""
训练指标全面诊断脚本
用法: python3 scripts/check_metrics.py
"""
import sys, os, glob, time
from datetime import datetime, timedelta
from pathlib import Path

# ── 配置 ──
STATE_FILE = '/data/mortal/mortal.pth'
BEST_FILE  = '/data/mortal/best.pth'
BASELINE_FILE = '/data/mortal/baseline.pth'
TB_DIR     = '/data/mortal/logs'
CHECKPOINT_DIR = '/data/mortal/checkpoints'
LOG_FILE   = '/tmp/blood-trainer.log'

def hr(title=""):
    print(f"\n{'═'*60}")
    if title:
        print(f"  {title}")
        print(f"{'═'*60}")

def fmt_time(ts):
    if ts is None: return "N/A"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def fmt_ago(ts):
    if ts is None: return ""
    delta = time.time() - ts
    if delta < 60: return f"({delta:.0f}s ago)"
    if delta < 3600: return f"({delta/60:.0f}m ago)"
    if delta < 86400: return f"({delta/3600:.1f}h ago)"
    return f"({delta/86400:.1f}d ago)"

# ══════════════════════════════════════════════════════
hr("1. CHECKPOINT 状态")
# ══════════════════════════════════════════════════════
try:
    import torch
    state = torch.load(STATE_FILE, map_location='cpu', weights_only=False)
    steps = state.get('steps', 0)
    best_perf = state.get('best_perf', {})
    ts = state.get('timestamp')
    config = state.get('config', {})
    
    print(f"  当前训练步数:  {steps:,}")
    print(f"  保存时间:      {fmt_time(ts)} {fmt_ago(ts)}")
    print(f"  最佳 avg_rank: {best_perf.get('avg_rank', 'N/A')}")
    print(f"  最佳 avg_pt:   {best_perf.get('avg_pt', 'N/A')}")
    
    # 学习率
    sched = state.get('scheduler', {})
    if '_last_lr' in sched:
        print(f"  当前学习率:    {sched['_last_lr'][0]:.6e}")
    
    # 阶段判断
    env_cfg = config.get('env', {})
    rs_cfg = config.get('reward_shaping', {})
    print(f"\n  配置快照:")
    print(f"    gamma:            {env_cfg.get('gamma', 'N/A')}")
    print(f"    td_lambda:        {env_cfg.get('td_lambda', 'N/A')}")
    print(f"    rank_bonus:       {'ON' if rs_cfg.get('rank_bonus_enabled') else 'OFF'}")
    print(f"    action_bonus:     {'ON' if rs_cfg.get('action_bonus_enabled') else 'OFF'}")
    print(f"    baseline_thresh:  {rs_cfg.get('baseline_update_threshold', 'N/A')}")
    
    tp_cfg = config.get('train_play', {}).get('default', {})
    print(f"    boltzmann_eps:    {tp_cfg.get('boltzmann_epsilon', 'N/A')}")
    print(f"    boltzmann_temp:   {tp_cfg.get('boltzmann_temp', 'N/A')}")

except FileNotFoundError:
    print(f"  ✗ 未找到: {STATE_FILE}")
    state = None; steps = 0; best_perf = {}; config = {}
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    state = None; steps = 0; best_perf = {}; config = {}

# ══════════════════════════════════════════════════════
hr("2. 文件状态")
# ══════════════════════════════════════════════════════
for label, path in [("mortal.pth", STATE_FILE), ("best.pth", BEST_FILE), ("baseline.pth", BASELINE_FILE)]:
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {label:15s}  {size_mb:6.1f} MB  {fmt_time(mtime)} {fmt_ago(mtime)}")
    else:
        print(f"  {label:15s}  ✗ 不存在")

# Checkpoints
if os.path.isdir(CHECKPOINT_DIR):
    ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/mortal_*.pth"))
    if ckpts:
        print(f"\n  历史 checkpoint ({len(ckpts)} 个):")
        for c in ckpts[-5:]:  # 只显示最近5个
            mtime = os.path.getmtime(c)
            print(f"    {os.path.basename(c):25s} {fmt_time(mtime)} {fmt_ago(mtime)}")
        if len(ckpts) > 5:
            print(f"    ... 还有 {len(ckpts)-5} 个更早的")

# ══════════════════════════════════════════════════════
hr("3. BASELINE 对比")
# ══════════════════════════════════════════════════════
baseline_update_threshold = config.get('reward_shaping', {}).get('baseline_update_threshold', 3.2)
avg_pt = best_perf.get('avg_pt', -999)
avg_rank = best_perf.get('avg_rank', 4.0)

print(f"  当前 best avg_pt:   {avg_pt}")
print(f"  更新阈值:           {baseline_update_threshold}")
print(f"  avg_pt >= 阈值?     {'✓ YES' if avg_pt >= baseline_update_threshold else '✗ NO'} ({avg_pt} vs {baseline_update_threshold})")

if os.path.exists(BASELINE_FILE) and os.path.exists(STATE_FILE):
    bl_mtime = os.path.getmtime(BASELINE_FILE)
    st_mtime = os.path.getmtime(STATE_FILE)
    gap = st_mtime - bl_mtime
    print(f"  baseline 年龄:      {fmt_time(bl_mtime)} {fmt_ago(bl_mtime)}")
    print(f"  mortal  年龄:       {fmt_time(st_mtime)} {fmt_ago(st_mtime)}")
    if gap > 3600:
        print(f"  ⚠️  baseline 比 mortal 旧 {gap/3600:.1f} 小时")
    
    # 尝试加载 baseline 步数
    try:
        bl_state = torch.load(BASELINE_FILE, map_location='cpu', weights_only=False)
        bl_steps = bl_state.get('steps', 0)
        print(f"  baseline 步数:      {bl_steps:,}")
        print(f"  mortal  步数:       {steps:,}")
        if steps > 0 and bl_steps > 0:
            print(f"  步数差距:           {steps - bl_steps:,} steps")
    except:
        pass

need_update = False
if avg_pt >= baseline_update_threshold:
    print(f"\n  ✓ 建议: avg_pt ({avg_pt:.4f}) 已达阈值 ({baseline_update_threshold})，可更新 baseline")
    need_update = True
elif steps > 0:
    gap_to_threshold = baseline_update_threshold - avg_pt
    print(f"\n  ✗ 尚未达到阈值，距离: {gap_to_threshold:.4f}")
    # 当 avg_rank 很高（接近4）时，说明模型还在随机水平
    if avg_rank > 3.5:
        print(f"  ℹ️  avg_rank={avg_rank:.4f} 接近随机(4.0)，模型尚处早期阶段")
    elif avg_rank > 2.8:
        print(f"  ℹ️  avg_rank={avg_rank:.4f} 已有一定区分度，继续训练")
    else:
        print(f"  ℹ️  avg_rank={avg_rank:.4f} 表现较好")

# ══════════════════════════════════════════════════════
hr("4. 训练日志 (最近)")
# ══════════════════════════════════════════════════════
log_candidates = [LOG_FILE, '/tmp/blood-trainer.log']
# Also check nohup.out in common locations
for d in ['/data/blood/mortal', '/data/blood', '/data/mortal']:
    log_candidates.append(f'{d}/nohup.out')

log_found = None
for lf in log_candidates:
    if os.path.exists(lf):
        log_found = lf
        break

if log_found:
    print(f"  日志文件: {log_found}")
    print(f"  大小: {os.path.getsize(log_found)/1024:.1f} KB")
    print(f"  最后修改: {fmt_time(os.path.getmtime(log_found))} {fmt_ago(os.path.getmtime(log_found))}")
    
    # 提取最近的关键指标
    print(f"\n  --- 最近的 test_play 结果 ---")
    lines = []
    with open(log_found, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if any(kw in line for kw in ['avg rank', 'avg pt', 'total steps', 'new record', 'Baseline updated', 'TRAIN']):
                lines.append(line)
    
    if lines:
        # 显示最近30条关键日志
        for line in lines[-30:]:
            print(f"    {line}")
    else:
        print("    (未找到 test_play 相关日志)")
    
    # 提取最近的 loss
    print(f"\n  --- 最近的训练进度 ---")
    progress_lines = []
    with open(log_found, 'r', errors='ignore') as f:
        for line in f:
            if 'total steps' in line or 'TRAIN' in line:
                progress_lines.append(line.strip())
    if progress_lines:
        for line in progress_lines[-5:]:
            print(f"    {line}")
    else:
        print("    (未找到训练进度日志)")
else:
    print("  ✗ 未找到训练日志文件")
    print("  尝试过的路径:")
    for lf in log_candidates:
        print(f"    {lf}")

# ══════════════════════════════════════════════════════
hr("5. TENSORBOARD 指标")
# ══════════════════════════════════════════════════════
if os.path.isdir(TB_DIR):
    event_files = glob.glob(f"{TB_DIR}/**/events.out.tfevents.*", recursive=True)
    if event_files:
        latest = max(event_files, key=os.path.getmtime)
        print(f"  TensorBoard 目录: {TB_DIR}")
        print(f"  事件文件数量:     {len(event_files)}")
        print(f"  最新事件文件:     {os.path.basename(latest)}")
        print(f"  最后修改:         {fmt_time(os.path.getmtime(latest))} {fmt_ago(os.path.getmtime(latest))}")
        
        # 尝试读取最近的 scalar 数据
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(TB_DIR, size_guidance={'scalars': 50})
            ea.Reload()
            
            tags = ea.Tags().get('scalars', [])
            print(f"  可用指标 ({len(tags)} 个): {', '.join(sorted(tags)[:20])}")
            if len(tags) > 20:
                print(f"    ... 还有 {len(tags)-20} 个")
            
            # 读取关键指标的最新值
            key_tags = [
                'test_play/avg_ranking', 'test_play/avg_pt',
                'loss/dqn_loss', 'loss/next_rank_loss', 'loss/ding_que_ce_loss', 'loss/opp_wait_loss',
                'ding_que/aux_match_rate', 'ding_que/dqn_match_rate',
                'hparam/lr',
            ]
            
            print(f"\n  --- 关键指标最新值 ---")
            for tag in key_tags:
                if tag in tags:
                    events = ea.Scalars(tag)
                    if events:
                        last = events[-1]
                        print(f"    {tag:35s}  {last.value:12.6f}  (step {last.step:,})")
            
            # test_play 历史趋势
            if 'test_play/avg_ranking' in tags:
                rankings = ea.Scalars('test_play/avg_ranking')
                pts_data = ea.Scalars('test_play/avg_pt') if 'test_play/avg_pt' in tags else []
                
                if len(rankings) >= 2:
                    print(f"\n  --- test_play 历史 (共 {len(rankings)} 次) ---")
                    print(f"    {'Step':>10s}  {'avg_rank':>10s}  {'avg_pt':>10s}  {'趋势':>6s}")
                    print(f"    {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}")
                    
                    pts_map = {e.step: e.value for e in pts_data} if pts_data else {}
                    prev_rank = None
                    for r in rankings[-10:]:  # 最近10次
                        pt_val = pts_map.get(r.step, float('nan'))
                        trend = ""
                        if prev_rank is not None:
                            if r.value < prev_rank: trend = "↑ 好"
                            elif r.value > prev_rank: trend = "↓ 差"
                            else: trend = "→"
                        prev_rank = r.value
                        print(f"    {r.step:>10,}  {r.value:>10.4f}  {pt_val:>10.4f}  {trend}")
            
            # Loss 趋势
            if 'loss/dqn_loss' in tags:
                losses = ea.Scalars('loss/dqn_loss')
                if len(losses) >= 2:
                    print(f"\n  --- DQN Loss 趋势 (共 {len(losses)} 个点) ---")
                    recent = losses[-5:]
                    early = losses[:5]
                    print(f"    早期 (step ~{early[0].step:,}): {sum(e.value for e in early)/len(early):.6f}")
                    print(f"    最近 (step ~{recent[-1].step:,}): {sum(e.value for e in recent)/len(recent):.6f}")
                    
                    if losses[-1].value < losses[0].value:
                        pct = (1 - losses[-1].value / max(losses[0].value, 1e-8)) * 100
                        print(f"    变化: ↓ {pct:.1f}% (loss 在下降，学习正常)")
                    else:
                        pct = (losses[-1].value / max(losses[0].value, 1e-8) - 1) * 100
                        print(f"    变化: ↑ {pct:.1f}% (loss 在上升，需要关注)")
                    
        except ImportError:
            print("\n  ⚠️  tensorboard 未安装，无法读取详细指标")
            print("     安装: pip install tensorboard")
            print("     或远程查看: tensorboard --logdir /data/mortal/logs --bind_all")
        except Exception as e:
            print(f"\n  ⚠️  读取 TensorBoard 失败: {e}")
    else:
        print(f"  ✗ {TB_DIR} 中无事件文件")
else:
    print(f"  ✗ TensorBoard 目录不存在: {TB_DIR}")

# ══════════════════════════════════════════════════════
hr("6. 综合诊断")
# ══════════════════════════════════════════════════════

if steps == 0:
    print("  ⚠️  训练步数为 0 或无法读取 checkpoint")
    print("  请确认训练是否已经启动")
elif steps < 5000:
    print(f"  ℹ️  训练仅 {steps:,} 步，尚未完成首次 test_play (每 5000 步)")
    print(f"  距下次 test_play: {5000 - steps % 5000:,} 步")
elif steps < 10000:
    print(f"  ℹ️  训练 {steps:,} 步，极早期阶段")
    print(f"  avg_rank 预期在 2.5-3.5 之间（随机 baseline 对手弱）")
    print(f"  baseline 更新: 暂不需要，等 avg_pt >= {baseline_update_threshold}")
else:
    print(f"  训练进度: {steps:,} 步 ({steps/1000:.0f}k)")
    if avg_rank < 2.0:
        print(f"  🏆 avg_rank={avg_rank:.4f} 表现优秀")
    elif avg_rank < 2.5:
        print(f"  ✓ avg_rank={avg_rank:.4f} 表现良好")
    elif avg_rank < 3.0:
        print(f"  → avg_rank={avg_rank:.4f} 中等水平")
    else:
        print(f"  ⚠️  avg_rank={avg_rank:.4f} 偏高，检查是否有问题")
    
    if need_update:
        print(f"\n  🔄 建议更新 baseline:")
        print(f"     cp {STATE_FILE} {BASELINE_FILE}")
    elif os.path.exists(BASELINE_FILE):
        bl_mtime = os.path.getmtime(BASELINE_FILE)
        hours_old = (time.time() - bl_mtime) / 3600
        if hours_old > 24 and steps > 20000:
            print(f"\n  ⚠️  baseline 已 {hours_old:.0f} 小时未更新")
            print(f"     如果 avg_rank 持续不变，考虑手动更新:")
            print(f"     cp {STATE_FILE} {BASELINE_FILE}")

print()
hr()
print("  完成。如需更详细分析，请将上面的输出贴回给我。")
print()
