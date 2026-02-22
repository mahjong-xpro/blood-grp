"""CLI arguments and default hyperparameters for Blood Mahjong."""

from argparse import ArgumentParser


def add_blood_args(parser: ArgumentParser):
    """Add Blood-specific command-line arguments."""
    p = parser

    # Model — 与 Rust consts.rs 保持一致 (NUM_STUDENT_CHANNELS=384, Bottleneck 256ch/20blocks)
    p.add_argument("--blood_obs_channels", type=int, default=384)
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
    p.add_argument("--aux_dingque_weight", type=float, default=1.0)
    p.add_argument("--aux_opp_waits_weight", type=float, default=0.1)

    # Warmup reward shaping
    p.add_argument("--warmup_reward_shaping", default=False, action="store_true")
    p.add_argument("--warmup_steps", type=int, default=2_000_000,
                    help="Env steps for warmup phase (reward shaping + RuleBot opponents)")
    p.add_argument("--warmup_dq_bonus", type=float, default=0.05,
                    help="Bonus for correct dingque selection during warmup")
    p.add_argument("--warmup_win_bonus", type=float, default=0.1,
                    help="Bonus for winning during warmup")
    p.add_argument("--warmup_deal_in_penalty", type=float, default=0.0,
                    help="Penalty for dealing-in during warmup (unused by default)")

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
        normalize_input=True,
        normalize_input_keys=["obs", "oracle_obs"],
        experiment="blood_v2",
    )
