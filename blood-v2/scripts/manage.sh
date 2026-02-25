#!/usr/bin/env bash
# manage.sh — Blood-v2 统一管理入口
#
# 用法:
#   ./scripts/manage.sh <命令> [选项]
#
# 命令:
#   train   <phase> [--device cuda] [--resume]   启动五阶段训练
#   train   pipeline [--device gpu] [--num-policies 1]  运行完整五阶段流水线
#   monitor [--port 6006]                         启动 TensorBoard 监控
#   eval    [--checkpoint <path>] [...]           运行评估
#   export  --checkpoint <path> [--quantize]      导出 ONNX 模型
#   status                                        显示训练状态（checkpoint 列表）
#   build                                         编译 Rust 引擎（maturin develop）
#   record  [--checkpoint <path>] [--games N]     录制回放文件
#   replay  [--log-dir <dir>] [--port 5001]       启动回放查看器
#   help                                          显示此帮助
#
# 五阶段训练流水线:
#   1. warmup              — RuleBot 对手, 无 LSTM (2.5M steps)
#   2. warmup_transition   — RuleBot 对手, LSTM 开启 (2.5M steps)
#   3. competitive         — Self-play (2.5M steps)
#   4. competitive_distill — Oracle Value 蒸馏 (2.5M steps)
#   5. elite               — RTPA+ISMCE 精英训练 (50M steps)
#
# 示例:
#   ./scripts/manage.sh train warmup
#   ./scripts/manage.sh train warmup_transition --resume
#   ./scripts/manage.sh train competitive --device cuda
#   ./scripts/manage.sh train distill
#   ./scripts/manage.sh train pipeline
#   ./scripts/manage.sh monitor
#   ./scripts/manage.sh eval --checkpoint checkpoints/blood_v2_elite/best
#   ./scripts/manage.sh export --checkpoint checkpoints/blood_v2_elite/best --quantize
#   ./scripts/manage.sh status
#   ./scripts/manage.sh build
#   ./scripts/manage.sh record --games 20
#   ./scripts/manage.sh replay

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[blood]${NC} $*"; }
ok()    { echo -e "${GREEN}[blood]${NC} $*"; }
warn()  { echo -e "${YELLOW}[blood]${NC} $*"; }
die()   { echo -e "${RED}[blood] ERROR:${NC} $*" >&2; exit 1; }

# ── 阶段名称规范化 ────────────────────────────────────────────────────────────
normalize_phase() {
    case "$1" in
        transition|warmup_transition|warmup-transition) echo "warmup_transition" ;;
        distill|competitive_distill|competitive-distill) echo "competitive_distill" ;;
        warmup|competitive|elite) echo "$1" ;;
        *) echo "" ;;
    esac
}

# ── Checkpoint 链式查找 ───────────────────────────────────────────────────────
# 从前一阶段查找最佳 checkpoint
find_best_checkpoint() {
    local phase="$1"
    local exp_name="blood_v2_${phase}"

    # SF2 stores checkpoints in train_dir/<experiment>/checkpoint_p0/
    local sf2_dir="train_dir/${exp_name}/checkpoint_p0"
    # Also check legacy path checkpoints/<experiment>/
    local legacy_dir="checkpoints/${exp_name}"

    local best_ckpt=""

    # Try SF2 directory first (primary)
    if [[ -d "$sf2_dir" ]]; then
        if [[ -f "${sf2_dir}/checkpoint_best.pth" ]]; then
            best_ckpt="${sf2_dir}/checkpoint_best.pth"
        else
            best_ckpt=$(find "$sf2_dir" -name "checkpoint_*.pth" 2>/dev/null | sort -V | tail -1)
        fi
    fi

    # Fallback to legacy checkpoints/ directory
    if [[ -z "$best_ckpt" && -d "$legacy_dir" ]]; then
        if [[ -f "${legacy_dir}/checkpoint_best.pth" ]]; then
            best_ckpt="${legacy_dir}/checkpoint_best.pth"
        else
            best_ckpt=$(find "$legacy_dir" -name "checkpoint_*.pth" 2>/dev/null | sort -V | tail -1)
        fi
    fi

    echo "$best_ckpt"
}

# 获取前一阶段名称（用于 checkpoint 链式传递）
get_prev_phase() {
    case "$1" in
        warmup) echo "" ;;
        warmup_transition) echo "warmup" ;;
        competitive) echo "warmup_transition" ;;
        competitive_distill) echo "competitive" ;;
        elite) echo "competitive_distill" ;;
        *) echo "" ;;
    esac
}

# ── 帮助 ──────────────────────────────────────────────────────────────────────
usage() {
cat <<EOF
Blood-v2 管理脚本

用法: $(basename "$0") <命令> [选项]

命令:
  train <phase> [选项]
      phase: warmup | warmup_transition (transition)
             competitive | competitive_distill (distill) | elite
             pipeline  — 运行完整五阶段流水线
      --device <gpu|cpu>    训练设备 (默认: gpu)
      --resume              从最新 checkpoint 恢复
      --num-policies <N>    多 GPU 策略数 (默认: 1, 每个 policy 占一张 GPU)

  monitor [选项]
      --port <port>         TensorBoard 端口 (默认: 6006)
      --bind <addr>         绑定地址 (默认: 0.0.0.0)

  eval [选项]
      --checkpoint <path>   checkpoint 路径 (默认: 自动选最新 elite)
      其余参数透传给 blood.eval.evaluate

  export [选项]
      --checkpoint <path>   checkpoint 路径 (必填)
      --output <path>       输出 ONNX 路径 (默认: blood_policy.onnx)
      --quantize            同时导出 INT8 量化版本

  build [选项]
      --release             编译 release 模式 (默认)
      --dev                 编译 debug 模式

  record [选项]
      --checkpoint <path>   checkpoint 路径 (默认: 自动选最新)
      --games <N>           录制局数 (默认: 20)
      --log-dir <dir>       回放文件输出目录 (默认: replays/)
      --baseline <mode>     对手模式: rulebot | random (默认: rulebot)

  replay [选项]
      --log-dir <dir>       回放文件目录 (默认: replays/)
      --port <port>         Web 服务端口 (默认: 5001)
      --host <addr>         绑定地址 (默认: 0.0.0.0)

  status
      显示各阶段 checkpoint 状态

  help
      显示此帮助

五阶段训练流水线:
  1. warmup              — RuleBot 对手, 无 LSTM (2.5M steps)
  2. warmup_transition   — RuleBot 对手, LSTM 开启 (2.5M steps)
  3. competitive         — Self-play (2.5M steps)
  4. competitive_distill — Oracle Value 蒸馏 (2.5M steps)
  5. elite               — RTPA+ISMCE 精英训练 (50M steps)

  阶段别名:
    transition  → warmup_transition
    distill     → competitive_distill

示例:
  $(basename "$0") train warmup
  $(basename "$0") train transition --resume
  $(basename "$0") train competitive --device cuda --num-policies 8
  $(basename "$0") train distill
  $(basename "$0") train pipeline
  $(basename "$0") monitor --port 6007
  $(basename "$0") eval --checkpoint checkpoints/blood_v2_elite/best
  $(basename "$0") export --checkpoint checkpoints/blood_v2_elite/best --quantize
  $(basename "$0") build
  $(basename "$0") record --games 50 --log-dir replays/
  $(basename "$0") replay --log-dir replays/ --port 5001
EOF
}

# ── do_train (核心训练逻辑，供 cmd_train 和 train_pipeline 调用) ──────────────
do_train() {
    local phase="$1"
    local device="$2"
    local num_policies="$3"
    local resume="$4"
    shift 4
    local extra_args=("$@")

    local config="configs/${phase}.yaml"
    [[ -f "$config" ]] || die "配置文件不存在: $config"

    # Reduce CUDA allocator fragmentation (helps when VRAM is nearly full)
    export PYTORCH_ALLOC_CONF=expandable_segments:True

    info "启动训练: phase=$phase  device=$device  num_policies=$num_policies"

    local cmd=(python -m blood.training.runner
        --config "$config"
        --device "$device"
        --num_policies "$num_policies"
    )

    # Checkpoint chaining: if not resuming, find previous phase's best checkpoint
    if [[ "$resume" -ne 1 ]]; then
        local prev_phase
        prev_phase=$(get_prev_phase "$phase")
        if [[ -n "$prev_phase" ]]; then
            local init_ckpt
            init_ckpt=$(find_best_checkpoint "$prev_phase")
            if [[ -n "$init_ckpt" ]]; then
                info "链式加载 ${prev_phase} checkpoint: ${init_ckpt}"
                cmd+=(--init_checkpoint_path "$init_ckpt")
            else
                warn "未找到 ${prev_phase} 的 checkpoint，从头开始训练"
            fi
        fi
    else
        info "从最新 checkpoint 恢复"
        cmd+=(--load_checkpoint_kind best)
    fi

    cmd+=("${extra_args[@]+"${extra_args[@]}"}")

    info "命令: ${cmd[*]}"
    "${cmd[@]}"
    return $?
}

# ── train ─────────────────────────────────────────────────────────────────────
cmd_train() {
    local phase="${1:-}"; shift || true

    # Handle pipeline subcommand
    if [[ "$phase" == "pipeline" ]]; then
        train_pipeline "$@"
        return $?
    fi

    local device="gpu"
    local resume=0
    local num_policies=1
    local extra_args=()

    # Normalize phase name
    local normalized
    normalized=$(normalize_phase "$phase")
    case "$normalized" in
        warmup|warmup_transition|competitive|competitive_distill|elite) phase="$normalized" ;;
        "") die "未知阶段: ${phase:-<空>} (可选: warmup | warmup_transition/transition | competitive | competitive_distill/distill | elite | pipeline)" ;;
    esac

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device)       device="$2"; shift 2 ;;
            --resume)       resume=1; shift ;;
            --num-policies) num_policies="$2"; shift 2 ;;
            *)              extra_args+=("$1"); shift ;;
        esac
    done

    do_train "$phase" "$device" "$num_policies" "$resume" "${extra_args[@]+"${extra_args[@]}"}"
    # do_train is the last command; use exec-like behavior for single phase
    exit $?
}

# ── train pipeline ────────────────────────────────────────────────────────────
train_pipeline() {
    local device="gpu"
    local num_policies=1

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device)       device="$2"; shift 2 ;;
            --num-policies) num_policies="$2"; shift 2 ;;
            *)              die "pipeline 未知参数: $1" ;;
        esac
    done

    local PHASES=("warmup" "warmup_transition" "competitive" "competitive_distill" "elite")
    local PHASE_NAMES=(
        "Phase 1: Warmup (RuleBot, no LSTM)"
        "Phase 1.5: Warmup Transition (RuleBot, LSTM ON)"
        "Phase 2a: Competitive (Self-play)"
        "Phase 2b: Competitive Distill (Oracle Value)"
        "Phase 3: Elite (RTPA+ISMCE, 50M steps)"
    )

    info "=========================================="
    info "  Blood-V2 Full Training Pipeline"
    info "  5 stages, ~57.5M total env steps"
    info "=========================================="

    for i in "${!PHASES[@]}"; do
        local phase="${PHASES[$i]}"
        local name="${PHASE_NAMES[$i]}"

        info ""
        info ">>> Starting ${name}"
        info "    Config: configs/${phase}.yaml"

        # Run training for this phase (never --resume in pipeline; chaining is automatic)
        do_train "$phase" "$device" "$num_policies" 0
        local exit_code=$?

        if [[ $exit_code -ne 0 ]]; then
            die "Phase ${phase} failed with exit code ${exit_code}\nPipeline stopped. Fix the issue and resume with:\n  ./scripts/manage.sh train ${phase} --resume"
        fi

        ok ">>> Completed ${name}"

        # Brief pause between stages
        if [[ $i -lt $((${#PHASES[@]} - 1)) ]]; then
            info "    Waiting 10s before next stage..."
            sleep 10
        fi
    done

    info ""
    info "=========================================="
    ok   "  Pipeline Complete!"
    info "  Best model: checkpoints/blood_v2_elite/"
    info "=========================================="
}

# ── monitor ───────────────────────────────────────────────────────────────────
cmd_monitor() {
    local port=6006
    local bind="0.0.0.0"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)  port="$2"; shift 2 ;;
            --bind)  bind="$2"; shift 2 ;;
            *)       die "monitor 未知参数: $1" ;;
        esac
    done

    command -v tensorboard &>/dev/null || die "tensorboard 未安装，请运行: pip install tensorboard"
    [[ -d "train_dir" ]] || warn "train_dir/ 不存在，TensorBoard 将等待日志写入"

    info "启动 TensorBoard: http://${bind}:${port}"
    exec tensorboard --logdir=train_dir/ --port="$port" --bind_all
}

# ── eval ──────────────────────────────────────────────────────────────────────
cmd_eval() {
    local checkpoint=""
    local extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --checkpoint) checkpoint="$2"; shift 2 ;;
            *)            extra_args+=("$1"); shift ;;
        esac
    done

    # 自动选最新 checkpoint (按五阶段优先级)
    if [[ -z "$checkpoint" ]]; then
        for phase in elite competitive_distill competitive warmup_transition warmup; do
            local candidate="checkpoints/blood_v2_${phase}"
            if [[ -d "$candidate" ]]; then
                checkpoint="$candidate"
                info "自动选择 checkpoint: $checkpoint"
                break
            fi
        done
        [[ -n "$checkpoint" ]] || die "未找到 checkpoint，请用 --checkpoint 指定路径"
    fi

    info "运行评估: checkpoint=$checkpoint"
    exec python -m blood.eval.evaluate \
        --checkpoint "$checkpoint" \
        "${extra_args[@]+"${extra_args[@]}"}"
}

# ── export ────────────────────────────────────────────────────────────────────
cmd_export() {
    local checkpoint=""
    local output="blood_policy.onnx"
    local quantize=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --checkpoint) checkpoint="$2"; shift 2 ;;
            --output)     output="$2"; shift 2 ;;
            --quantize)   quantize=1; shift ;;
            *)            die "export 未知参数: $1" ;;
        esac
    done

    [[ -n "$checkpoint" ]] || die "export 需要 --checkpoint <path>"

    local cmd=(python scripts/export_onnx.py
        --checkpoint "$checkpoint"
        --output "$output"
    )
    [[ $quantize -eq 1 ]] && cmd+=(--quantize)

    info "导出 ONNX: $checkpoint → $output"
    exec "${cmd[@]}"
}

# ── build ─────────────────────────────────────────────────────────────────────
cmd_build() {
    local mode="--release"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dev)     mode=""; shift ;;
            --release) mode="--release"; shift ;;
            *)         die "build 未知参数: $1" ;;
        esac
    done

    command -v maturin &>/dev/null || die "maturin 未安装，请运行: pip install maturin"

    info "编译 Rust 引擎 (maturin develop ${mode})..."
    # shellcheck disable=SC2086
    maturin develop $mode
    ok "编译完成"
}

# ── record ────────────────────────────────────────────────────────────────────
cmd_record() {
    local checkpoint=""
    local games=20
    local log_dir="replays"
    local baseline="rulebot"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --checkpoint) checkpoint="$2"; shift 2 ;;
            --games)      games="$2"; shift 2 ;;
            --log-dir)    log_dir="$2"; shift 2 ;;
            --baseline)   baseline="$2"; shift 2 ;;
            *)            die "record 未知参数: $1" ;;
        esac
    done

    # 自动选最新 checkpoint (按五阶段优先级)
    if [[ -z "$checkpoint" ]]; then
        for phase in elite competitive_distill competitive warmup_transition warmup; do
            local dir="train_dir/blood_v2_${phase}/checkpoint_p0"
            if [[ -d "$dir" ]]; then
                # 选最新的 .pth 文件
                checkpoint=$(find "$dir" -name "checkpoint_*.pth" | sort -V | tail -1)
                if [[ -n "$checkpoint" ]]; then
                    info "自动选择 checkpoint: $checkpoint"
                    break
                fi
            fi
        done
        [[ -n "$checkpoint" ]] || die "未找到 checkpoint，请用 --checkpoint 指定路径"
    fi

    [[ -f "$checkpoint" ]] || die "checkpoint 文件不存在: $checkpoint"

    mkdir -p "$log_dir"
    info "录制 ${games} 局回放 → ${log_dir}/"
    info "checkpoint: $checkpoint  baseline: $baseline"

    exec python -m blood.eval.evaluate \
        --checkpoint "$checkpoint" \
        --num_games "$games" \
        --baseline "$baseline" \
        --save_replays "$log_dir"
}

# ── replay ────────────────────────────────────────────────────────────────────
cmd_replay() {
    local log_dir="replays"
    local port=5001
    local host="0.0.0.0"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --log-dir) log_dir="$2"; shift 2 ;;
            --port)    port="$2"; shift 2 ;;
            --host)    host="$2"; shift 2 ;;
            *)         die "replay 未知参数: $1" ;;
        esac
    done

    local viewer="$PROJECT_DIR/log-viewer/app.py"
    [[ -f "$viewer" ]] || die "回放查看器不存在: $viewer"

    if [[ ! -d "$log_dir" ]]; then
        warn "回放目录不存在: $log_dir，将自动创建"
        mkdir -p "$log_dir"
    fi

    local count
    count=$(find "$log_dir" -maxdepth 2 -name "*.json" -o -name "*.json.gz" 2>/dev/null | wc -l | tr -d ' ')
    info "回放目录: $log_dir  (${count} 个文件)"
    info "启动回放查看器: http://${host}:${port}"
    info "按 Ctrl+C 停止"

    exec python "$viewer" \
        --host "$host" \
        --port "$port" \
        --log-dir "$log_dir"
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
    echo ""
    echo "── Checkpoint 状态 ──────────────────────────────────────"
    for phase in warmup warmup_transition competitive competitive_distill elite; do
        local dir="checkpoints/blood_v2_${phase}"
        if [[ -d "$dir" ]]; then
            local count latest_step
            count=$(find "$dir" -name "*.pth" -o -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
            # Extract latest step number from checkpoint filenames
            latest_step=$(ls -1 "${dir}"/checkpoint_*.pth 2>/dev/null | sed 's/.*checkpoint_//' | sed 's/\.pth//' | sort -n | tail -1)
            if [[ -n "$latest_step" && "$latest_step" != "best" ]]; then
                ok "  ${phase}: $dir  (${count} 个权重文件, latest step: ${latest_step})"
            else
                ok "  ${phase}: $dir  (${count} 个权重文件)"
            fi
        else
            warn "  ${phase}: 未找到 ($dir)"
        fi
    done

    echo ""
    echo "── 训练进程 ─────────────────────────────────────────────"
    local found_running=0
    for phase in warmup warmup_transition competitive competitive_distill elite; do
        if pgrep -f "blood.training.runner.*${phase}" &>/dev/null; then
            ok "  ${phase}: 运行中"
            found_running=1
        fi
    done
    if [[ $found_running -eq 0 ]]; then
        warn "  无训练进程运行"
    fi

    echo ""
    echo "── 联赛池 ───────────────────────────────────────────────"
    local league_dir="checkpoints/league"
    if [[ -d "$league_dir" ]]; then
        local count
        count=$(find "$league_dir" -maxdepth 1 -name "*.pth" -o -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
        ok "  league: $league_dir  (${count} 个历史模型)"
    else
        warn "  league: 未找到 ($league_dir)"
    fi

    echo ""
    echo "── TensorBoard 日志 ─────────────────────────────────────"
    if [[ -d "train_dir" ]]; then
        local size
        size=$(du -sh train_dir 2>/dev/null | cut -f1)
        ok "  train_dir/  ($size)"
    else
        warn "  train_dir/ 不存在"
    fi

    echo ""
    echo "── 回放文件 ─────────────────────────────────────────────"
    if [[ -d "replays" ]]; then
        local count
        count=$(find replays -name "*.json" -o -name "*.json.gz" 2>/dev/null | wc -l | tr -d ' ')
        ok "  replays/  (${count} 个回放文件)"
    else
        warn "  replays/ 不存在 (运行 record 命令生成)"
    fi
    echo ""
}

# ── 主入口 ────────────────────────────────────────────────────────────────────
CMD="${1:-help}"; shift || true

case "$CMD" in
    train)   cmd_train   "$@" ;;
    monitor) cmd_monitor "$@" ;;
    eval)    cmd_eval    "$@" ;;
    export)  cmd_export  "$@" ;;
    build)   cmd_build   "$@" ;;
    record)  cmd_record  "$@" ;;
    replay)  cmd_replay  "$@" ;;
    status)  cmd_status  "$@" ;;
    help|-h|--help) usage ;;
    *) die "未知命令: $CMD  (运行 $(basename "$0") help 查看帮助)" ;;
esac
