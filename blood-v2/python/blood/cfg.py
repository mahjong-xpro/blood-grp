"""CLI arguments and default hyperparameters for Blood Mahjong."""

from argparse import ArgumentParser


def add_blood_args(parser: ArgumentParser):
    """Add Blood-specific command-line arguments."""
    p = parser

    # Model — 与 Rust consts.rs 保持一致 (NUM_STUDENT_CHANNELS=464, Bottleneck 256ch/20blocks)
    p.add_argument("--blood_obs_channels", type=int, default=464)
    p.add_argument("--blood_conv_channels", type=int, default=256)
    p.add_argument("--blood_num_res_blocks", type=int, default=20)
    p.add_argument("--blood_encoder_out_dim", type=int, default=1024)

    # Oracle
    p.add_argument("--oracle_enabled", default=True, type=lambda x: x.lower() != "false")
    p.add_argument("--no_oracle", dest="oracle_enabled", action="store_false")
    p.add_argument("--oracle_num_blocks", type=int, default=25)
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
    p.add_argument("--league_newest_weight", type=float, default=3.0)
    p.add_argument("--opponent_mode", type=str, default="rulebot",
                    choices=["rulebot", "selfplay", "random"])
    p.add_argument("--opponent_refresh_every", type=int, default=20,
                    help="Reload opponent model every N episodes")

    # Augmentation
    p.add_argument("--suit_augment_prob", type=float, default=0.5)

    # Auxiliary tasks
    p.add_argument("--aux_shanten_weight", type=float, default=1.0)
    p.add_argument("--aux_opp_waits_weight", type=float, default=0.3)

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
        normalize_input_keys=["obs", "oracle_obs"],
        experiment="blood_v2",
        use_rnn=True,
        rnn_type="lstm",
        rnn_size=1024,
        rnn_num_layers=1,
        rollout=32,
        recurrence=32,
    )
