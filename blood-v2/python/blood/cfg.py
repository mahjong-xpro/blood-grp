"""CLI arguments and default hyperparameters for Blood Mahjong."""

from argparse import ArgumentParser


def add_blood_args(parser: ArgumentParser):
    """Add Blood-specific command-line arguments."""
    p = parser

    # Game rules
    p.add_argument("--initial_score", type=int, default=100_000,
                    help="Initial score per player (blood mahjong: 100000 or 60000)")

    # Model — 与 Rust consts.rs 保持一致 (NUM_STUDENT_CHANNELS=470, Bottleneck 256ch/20blocks)
    p.add_argument("--blood_obs_channels", type=int, default=470)
    p.add_argument("--blood_conv_channels", type=int, default=256)
    p.add_argument("--blood_num_res_blocks", type=int, default=20)
    p.add_argument("--blood_encoder_out_dim", type=int, default=1024)
    p.add_argument("--blood_enc_proj_layers", type=int, default=1,
                    help="enc_proj 层数: 1=单层Linear(旧行为), 2=渐进压缩MLP(缓解信息瓶颈)")

    # TileAttention — architecture params (must be identical across ALL training stages)
    p.add_argument("--blood_num_tile_attn_layers", type=int, default=2,
                    help="TileAttention segment count (2-6). Residual blocks are evenly "
                         "distributed across segments, each followed by a TileAttention layer.")
    p.add_argument("--blood_tile_attn_heads", type=int, default=4,
                    help="TileAttention 注意力头数: 4=旧行为, 8=增强多模式跨花色交互")

    # Oracle
    p.add_argument("--oracle_enabled", default=True, type=lambda x: x.lower() != "false")
    p.add_argument("--no_oracle", dest="oracle_enabled", action="store_false")
    # 修复: 默认值从 25 改为 20，与所有阶段 yaml 配置保持一致
    p.add_argument("--oracle_num_blocks", type=int, default=20)
    p.add_argument("--oracle_num_tile_attn_layers", type=int, default=2,
                    help="Oracle TileAttention segment count (default 2, must match across stages)")
    p.add_argument("--oracle_tile_attn_heads", type=int, default=4,
                    help="Oracle TileAttention attention heads (default 4)")
    p.add_argument("--oracle_distill_weight", type=float, default=0.05)
    p.add_argument("--oracle_distill_temperature", type=float, default=2.0)
    p.add_argument("--oracle_ce_weight", type=float, default=0.1,
                    help="Weight for Oracle CE supervised loss")
    p.add_argument("--oracle_value_distill_weight", type=float, default=0.0,
                    help="Weight for Oracle value distillation (student critic → Oracle value). "
                         "Only effective after oracle_value_warmup_steps.")
    p.add_argument("--oracle_value_head_loss_weight", type=float, default=1.0,
                    help="Weight for oracle value head supervised loss against GAE returns. "
                         "Trains oracle value head before distillation is enabled.")
    p.add_argument("--oracle_value_warmup_steps", type=int, default=500_000,
                    help="Env steps before oracle value distillation activates. "
                         "Oracle value head must converge first.")

    # League / self-play
    p.add_argument("--league_enabled", default=True, action="store_true")
    p.add_argument("--no_league", dest="league_enabled", action="store_false")
    p.add_argument("--league_pool_dir", type=str, default="checkpoints/league/")
    p.add_argument("--league_add_every", type=int, default=50000)
    p.add_argument("--league_newest_weight", type=float, default=2.0,
                    help="多项式衰减指数 α，从 3.0 降到 2.0 提高有效多样性")
    p.add_argument("--league_uniform_floor", type=float, default=0.1,
                    help="最低采样概率下限，确保旧 checkpoint 也有一定概率被采样")
    p.add_argument("--league_self_play_prob", type=float, default=0.2,
                    help="使用当前最新策略自博弈的概率（不从历史池采样）")
    p.add_argument("--opponent_mode", type=str, default="rulebot",
                    choices=["rulebot", "selfplay", "random"])
    p.add_argument("--opponent_refresh_every", type=int, default=20,
                    help="Reload opponent model every N episodes")

    # Elo rating system
    p.add_argument("--blood_elo_enabled", default=True,
                    type=lambda x: str(x).lower() != "false",
                    help="Enable persistent Elo rating tracking across training")
    p.add_argument("--blood_elo_k_base", default=32.0, type=float,
                    help="Base K-factor for established players (>=new_threshold games)")
    p.add_argument("--blood_elo_k_new", default=64.0, type=float,
                    help="K-factor for new players (<new_threshold games)")
    p.add_argument("--blood_elo_new_threshold", default=30, type=int,
                    help="Game count below which a player is considered 'new' (higher K)")
    p.add_argument("--blood_elo_sampling", default=False,
                    type=lambda x: str(x).lower() != "false",
                    help="Use Elo-weighted Gaussian opponent sampling in league")
    p.add_argument("--blood_elo_sampling_sigma", default=200.0, type=float,
                    help="Gaussian sigma for Elo-weighted opponent sampling (Elo spread)")

    # Augmentation
    p.add_argument("--suit_augment_prob", type=float, default=0.5)

    # Auxiliary tasks
    p.add_argument("--aux_shanten_weight", type=float, default=1.0)
    p.add_argument("--aux_opp_waits_weight", type=float, default=0.3)
    p.add_argument("--aux_focal_alpha", type=float, default=0.25,
                    help="Focal Loss alpha: 正样本权重因子，缓解听牌预测类别不平衡")
    p.add_argument("--aux_focal_gamma", type=float, default=2.0,
                    help="Focal Loss gamma: 聚焦参数，越大越关注难分类样本")

    # Warmup reward shaping
    p.add_argument("--warmup_reward_shaping", default=False, action="store_true")
    p.add_argument("--warmup_dq_bonus", type=float, default=0.05,
                    help="Bonus for correct dingque selection during warmup")
    p.add_argument("--warmup_win_bonus", type=float, default=0.1,
                    help="Bonus for winning during warmup")
    p.add_argument("--warmup_deal_in_penalty", type=float, default=0.0,
                    help="Penalty for dealing-in during warmup (unused by default)")
    p.add_argument("--warmup_dangerous_discard_penalty", type=float, default=0.03,
                    help="Penalty for discarding a tile an opponent is waiting for (oracle-guided, warmup only)")

    # Structured reward shaping (all phases)
    # Calibrated for REWARD_NORM=32000 (max single-player payment per hand, 6-fan ron cap).
    # agent_delta can reach 96000 (6-fan tsumo × 3 payers). Sqrt-compressed range:
    # max ron=+1.0, max tsumo=+1.732 (sqrt(3)); 1-fan ron=+0.177, 1-fan tsumo=+0.306.
    p.add_argument("--reward_tsumo_bonus", type=float, default=0.1,
                    help="Extra reward for tsumo win (≥2 opponents paid)")
    p.add_argument("--reward_deal_in_penalty", type=float, default=0.05,
                    help="Extra penalty for dealing in (≥1 opponent gains, agent loses)")
    p.add_argument("--reward_shanten_progress", type=float, default=0.003,
                    help="Reward per shanten reduction (dense progress signal)")
    p.add_argument("--reward_shanten_regress", type=float, default=0.001,
                    help="Penalty per shanten increase")
    p.add_argument("--shanten_reward_decay_steps", type=int, default=0,
                    help="向听奖励线性衰减步数 (0=不衰减)。在此步数内从原始值衰减到 min_ratio 倍")
    p.add_argument("--shanten_reward_min_ratio", type=float, default=0.3,
                    help="向听奖励衰减下限比例 (默认0.3=衰减到原始值的30%%)")
    p.add_argument("--shanten_fan_bonus_scale", type=float, default=0.3,
                    help="向听奖励番数加权缩放因子 (0=禁用)。向听改善时乘以 "
                         "(1 + scale * estimated_fan / max_fan)，引导模型追求高番手牌")
    p.add_argument("--shanten_fan_max", type=float, default=8.0,
                    help="番数归一化上限。血战麻将理论最高6番(封顶)，"
                         "设为8.0留出余量避免加权系数过大")
    p.add_argument("--reward_safe_discard", type=float, default=0.0,
                    help="Reward for discarding a safe tile when any opponent is tenpai")
    p.add_argument("--reward_rank_bonus", type=float, default=0.0,
                    help="Rank bonus at game end: 1st=+bonus, 2nd=+0.3×bonus, 3rd=-0.3×bonus, 4th=-bonus")

    # RTPA (Runtime Policy Adaptation)
    p.add_argument("--rtpa_enabled", default=False, action="store_true")
    p.add_argument("--rtpa_attack_temp", type=float, default=0.8,
                    help="Temperature when tenpai (aggressive)")
    p.add_argument("--rtpa_defend_temp", type=float, default=1.5,
                    help="Temperature when opponents are tenpai (defensive)")

    # ISMCE
    p.add_argument("--ismce_enabled", default=False, action="store_true")
    p.add_argument("--ismce_num_worlds", type=int, default=64,
                    help="Number of world samples for ISMCE")
    p.add_argument("--ismce_rollout_depth", type=int, default=4,
                    help="Rollout depth for ISMCE playout")

    # Advantage clipping — clip advantages to [-adv_clip, adv_clip] before PPO update.
    # Prevents extreme advantage samples (observed ±4.7 in warmup) from dominating gradients.
    # Set to 0 to disable.
    p.add_argument("--adv_clip", type=float, default=0.0,
                    help="Clip advantages to [-adv_clip, adv_clip] before PPO update (0=disabled)")

    # Monitoring throttle
    p.add_argument("--blood_metrics_interval", default=100, type=int,
                    help="Compute expensive monitoring metrics every N minibatches")

    # Cross-phase checkpoint chaining
    p.add_argument("--init_checkpoint_path", type=str, default="",
                    help="Path to a checkpoint from a previous training phase. "
                         "Model weights are seeded into the new experiment directory; "
                         "optimizer state is reset so the new phase trains from scratch.")

    # Dynamic hyperparameter schedules (within a training stage)
    p.add_argument("--blood_schedule_entropy", default="", type=str,
                    help="Entropy coeff schedule: 'type,start,end,start_step,end_step'")
    p.add_argument("--blood_schedule_adv_clip", default="", type=str,
                    help="Advantage clip schedule: 'type,start,end,start_step,end_step'")
    p.add_argument("--blood_schedule_extra", default="", type=str,
                    help="Extra schedules: 'param:type,args;param:type,args'")


def blood_override_defaults(parser: ArgumentParser):
    """Override Sample Factory defaults for Blood Mahjong."""
    parser.set_defaults(
        env="blood_mahjong",
        encoder_custom="blood_encoder",
        num_workers=8,
        num_envs_per_worker=32,
        batch_size=8192,
        num_batches_per_epoch=4,
        ppo_clip_ratio=0.1,
        max_grad_norm=1.0,
        gamma=0.998,
        gae_lambda=0.95,
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_minibatch",
        exploration_loss_coeff=0.005,
        value_loss_coeff=1.0,
        normalize_input=True,
        normalize_input_keys=["obs"],
        experiment="blood_v2",
        use_rnn=True,
        rnn_type="lstm",
        rnn_size=512,
        rnn_num_layers=2,
        rollout=32,
        recurrence=32,
    )
