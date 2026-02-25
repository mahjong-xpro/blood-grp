"""Runtime Policy Adaptation (RTPA).

根据游戏状态动态调整策略温度：
- 听牌时 → 降低温度（进攻，利用和牌机会）
- 对手可能听牌时 → 提高温度（防守，多样化出牌）
- 分数落后时 → 略微激进
- 分数领先时 → 略微保守
"""

import numpy as np

from blood.consts import (
    NUM_TILE_TYPES, ACTION_SPACE, NUM_STUDENT_CHANNELS,
    INITIAL_SCORE, REWARD_NORM, MAX_TURNS,
    CH_WALL_REMAINING, CH_OPP_MELD_BASE, CH_SHANTEN_BASE,
    CH_TURN_PROGRESS, CH_OPP_AGARI_BASE,
    CH_OPP_KAWA_BASE, CH_OPP_SUIT_RATIO_BASE,
    CH_OPP_TERMINAL_RATIO_BASE, CH_SELF_DISCARD_COUNT,
)

# ── 对手牌河通道布局常量 ──────────────────────────────────────────────────────
# Section 6 中每个对手占 58 个通道：28×2(位置编码) + 1(衰减) + 1(摸切衰减)
_OPP_KAWA_STRIDE = MAX_TURNS * 2 + 2  # 58


class RTPA:
    """推理时策略自适应（Runtime Policy Adaptation）。"""

    def __init__(
        self,
        base_temp: float = 1.0,
        attack_temp: float = 0.8,
        defend_temp: float = 1.5,
        score_sensitivity: float = 0.1,
    ):
        self.base_temp = base_temp
        self.attack_temp = attack_temp
        self.defend_temp = defend_temp
        self.score_sensitivity = score_sensitivity

    def compute_temperature(
        self,
        is_tenpai: bool,
        opponents_likely_tenpai: int,
        my_score: int,
        avg_opponent_score: float,
        wall_remaining: int,
    ) -> float:
        """根据游戏上下文计算自适应温度。"""
        temp = self.base_temp

        if is_tenpai:
            temp = self.attack_temp
        elif opponents_likely_tenpai > 0:
            # 根据危险对手数量缩放防守力度
            defense_factor = min(opponents_likely_tenpai, 3) / 3.0
            temp = self.base_temp + defense_factor * (self.defend_temp - self.base_temp)

        # 领先时（score_diff > 0）：提高温度 → 保守
        # 落后时（score_diff < 0）：降低温度 → 激进
        score_diff = my_score - avg_opponent_score
        score_adjust = self.score_sensitivity * np.sign(score_diff) * min(abs(score_diff) / float(REWARD_NORM), 0.2)
        temp += score_adjust

        # 残局放大：从 wall_remaining=20 开始线性渐变到 wall_remaining=0
        # 替代原来 wall_remaining<10 的阶跃函数，提供更平滑的过渡
        if wall_remaining < 20:
            # 线性插值：wall=20 时倍率=1.0，wall=0 时倍率=1.3
            endgame_factor = 1.0 + 0.3 * (1.0 - wall_remaining / 20.0)
            temp *= endgame_factor

        return max(0.3, min(temp, 3.0))

    def adapt_logits(
        self,
        logits: np.ndarray,
        mask: np.ndarray,
        is_tenpai: bool = False,
        opponents_likely_tenpai: int = 0,
        my_score: int = 100000,
        avg_opponent_score: float = 100000.0,
        wall_remaining: int = 50,
        danger_scores: np.ndarray = None,
    ) -> np.ndarray:
        """对策略 logits 应用 RTPA。

        返回经过温度调整和可选危险惩罚后的 logits。
        """
        temp = self.compute_temperature(
            is_tenpai, opponents_likely_tenpai,
            my_score, avg_opponent_score, wall_remaining,
        )

        adjusted = logits.copy()
        adjusted[mask < 0.5] = -1e9

        if danger_scores is not None and not is_tenpai and opponents_likely_tenpai > 0:
            for i in range(NUM_TILE_TYPES):
                if mask[i] > 0.5:
                    adjusted[i] -= danger_scores[i] * 2.0

        adjusted /= max(temp, 1e-8)
        return adjusted


class GameStateTracker:
    """跟踪 RTPA 决策所需的游戏状态特征。

    从 470×27 的学生观测张量中提取多维信号，综合判断对手听牌概率，
    替代原来仅依赖副露比例（meld_ratio >= 0.5）的粗糙启发式。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.my_tenpai = False
        self.opponents_tenpai_count = 0
        self.my_score = INITIAL_SCORE
        self.opponent_scores = [INITIAL_SCORE] * 3
        self.wall_remaining = 108 - 13 * 4

    def update_from_obs(self, obs: np.ndarray, scores: list = None):
        """从观测张量中提取游戏状态特征。

        解析 464×27 学生观测中的已知通道偏移量
        （来源：crates/engine/src/obs/student.rs）：

        自家状态：
        - CH_WALL_REMAINING (35): wall_remaining / 55.0
        - CH_SHANTEN_BASE (341-345): 向听数 one-hot，ch341=听牌

        对手听牌推断（多信号综合）：
        - CH_OPP_MELD_BASE (333-335): 对手副露数 / MAX_MELDS
        - CH_OPP_AGARI_BASE (32-34): 对手是否已和牌
        - CH_OPP_KAWA_BASE (98+): 对手牌河（摸切模式变化）
        - CH_OPP_SUIT_RATIO_BASE (320-328): 对手花色打牌比例
        - CH_OPP_TERMINAL_RATIO_BASE (336-338): 对手幺九打牌比例
        - CH_TURN_PROGRESS (17): 回合进度
        """
        if scores is not None and len(scores) >= 4:
            self.my_score = scores[0]
            self.opponent_scores = list(scores[1:4])

        if obs is None or obs.shape[0] < NUM_STUDENT_CHANNELS * NUM_TILE_TYPES:
            return

        obs_2d = obs.reshape(-1, NUM_TILE_TYPES)

        # ── 提取牌墙剩余数 ──
        if obs_2d.shape[0] > CH_WALL_REMAINING:
            wall_val = float(obs_2d[CH_WALL_REMAINING].mean())
            self.wall_remaining = max(0, int(wall_val * 55.0 + 0.5))

        # ── 提取自家向听数（听牌判断）──
        if obs_2d.shape[0] > CH_SHANTEN_BASE + 4:
            shanten_channels = [float(obs_2d[CH_SHANTEN_BASE + i].mean()) for i in range(5)]
            self.my_tenpai = shanten_channels[0] > 0.5

        # ── 提取回合进度 ──
        turn_progress = 0.0
        if obs_2d.shape[0] > CH_TURN_PROGRESS:
            turn_progress = float(obs_2d[CH_TURN_PROGRESS].mean())

        # ── 综合推断对手听牌 ──
        self.opponents_tenpai_count = 0
        for i in range(3):
            tenpai_score = self._estimate_opponent_tenpai(obs_2d, i, turn_progress)
            if tenpai_score >= 0.5:
                self.opponents_tenpai_count += 1

    def _estimate_opponent_tenpai(
        self, obs_2d: np.ndarray, opp_idx: int, turn_progress: float
    ) -> float:
        """综合多维信号估算单个对手的听牌概率。

        信号权重：
        1. 副露数（0.25）：副露越多，手牌越少，越可能听牌
        2. 摸切率变化（0.25）：听牌后倾向于摸切（摸什么打什么）
        3. 幺九打牌比例（0.15）：清一色/断幺九等牌型的间接信号
        4. 花色集中度（0.15）：打牌花色高度集中暗示定缺已完成且手牌成型
        5. 牌河长度（0.10）：打牌越多，越可能已经听牌
        6. 已和牌排除（-∞）：已和牌的对手不再构成威胁

        返回 [0, 1] 范围的听牌概率估计。
        """
        # ── 检查对手是否已和牌（已和牌则排除）──
        if obs_2d.shape[0] > CH_OPP_AGARI_BASE + opp_idx:
            agari = float(obs_2d[CH_OPP_AGARI_BASE + opp_idx].mean())
            if agari > 0.5:
                return 0.0

        score = 0.0

        # ── 信号1：副露数（权重 0.25）──
        # 副露数 / MAX_MELDS，值域 [0, 1]
        # 副露 >= 2 时（ratio >= 0.5）给予较高分数
        if obs_2d.shape[0] > CH_OPP_MELD_BASE + opp_idx:
            meld_ratio = float(obs_2d[CH_OPP_MELD_BASE + opp_idx].mean())
            # 使用 sigmoid 风格的平滑映射：0副露→0, 1副露→0.3, 2副露→0.7, 3+→1.0
            meld_signal = min(meld_ratio * 2.0, 1.0)
            score += 0.25 * meld_signal

        # ── 信号2：摸切率（权重 0.25）──
        # 从对手牌河的摸切衰减通道提取近期摸切比例
        # 听牌后玩家倾向于摸切（不需要的牌直接打出）
        opp_kawa_start = CH_OPP_KAWA_BASE + opp_idx * _OPP_KAWA_STRIDE
        tsumogiri_decay_ch = opp_kawa_start + MAX_TURNS * 2 + 1  # 摸切衰减通道
        discard_decay_ch = opp_kawa_start + MAX_TURNS * 2        # 打牌衰减通道
        if obs_2d.shape[0] > tsumogiri_decay_ch:
            tg_sum = float(obs_2d[tsumogiri_decay_ch].sum())
            disc_sum = float(obs_2d[discard_decay_ch].sum())
            if disc_sum > 0.1:
                # 摸切占总打牌的比例（衰减加权）
                tsumogiri_ratio = tg_sum / disc_sum
                # 摸切率 > 0.5 是强听牌信号
                tg_signal = min(max(tsumogiri_ratio - 0.2, 0.0) / 0.6, 1.0)
                score += 0.25 * tg_signal

        # ── 信号3：幺九打牌比例（权重 0.15）──
        # 高幺九打牌比例暗示对手在做断幺九或清一色
        if obs_2d.shape[0] > CH_OPP_TERMINAL_RATIO_BASE + opp_idx:
            terminal_ratio = float(obs_2d[CH_OPP_TERMINAL_RATIO_BASE + opp_idx].mean())
            # 幺九比例 > 0.4 时开始给分
            terminal_signal = min(max(terminal_ratio - 0.3, 0.0) / 0.4, 1.0)
            score += 0.15 * terminal_signal

        # ── 信号4：花色集中度（权重 0.15）──
        # 对手打牌花色高度集中 → 定缺完成且手牌成型
        suit_ratio_start = CH_OPP_SUIT_RATIO_BASE + opp_idx * 3
        if obs_2d.shape[0] > suit_ratio_start + 2:
            suit_ratios = [
                float(obs_2d[suit_ratio_start + s].mean()) for s in range(3)
            ]
            max_suit_ratio = max(suit_ratios)
            # 某花色打牌比例 > 0.6 说明定缺花色集中打出，手牌趋于成型
            concentration_signal = min(max(max_suit_ratio - 0.4, 0.0) / 0.4, 1.0)
            score += 0.15 * concentration_signal

        # ── 信号5：牌河长度 / 回合进度（权重 0.10）──
        # 游戏越深入，对手听牌的先验概率越高
        progress_signal = min(turn_progress / 0.7, 1.0)
        score += 0.10 * progress_signal

        # ── 信号6：回合进度加成 ──
        # 后半局（turn_progress > 0.5）时，整体听牌概率上调
        # 这反映了随着游戏进行，所有玩家趋向听牌的自然趋势
        if turn_progress > 0.5:
            late_game_boost = 0.10 * (turn_progress - 0.5) / 0.5
            score += late_game_boost

        return min(score, 1.0)

    @property
    def avg_opponent_score(self) -> float:
        return sum(self.opponent_scores) / max(len(self.opponent_scores), 1)
