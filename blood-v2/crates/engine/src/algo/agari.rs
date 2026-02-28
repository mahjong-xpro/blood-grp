use crate::consts::*;
use crate::tile::{Tile, Suit, is_terminal};
use crate::hand::{HandCounts, MeldType};
use crate::state::ding_que;

/// Configurable fan rules
#[derive(Debug, Clone, Copy)]
pub struct FanConfig {
    pub menqing: bool,
    pub duanyaojiu: bool,
    pub daiyaojiu: bool,
    pub yitiaolong: bool,
    pub jiaxinwu: bool,
    pub haidi: bool,
    pub tianhu_dihu: bool,
}

impl Default for FanConfig {
    fn default() -> Self {
        Self {
            menqing: true,
            duanyaojiu: true,
            daiyaojiu: true,
            yitiaolong: true,
            jiaxinwu: true,
            haidi: true,
            tianhu_dihu: true,
        }
    }
}

/// Context for calculating fan on a winning hand
#[derive(Debug, Clone)]
pub struct WinContext {
    pub tehai: HandCounts,       // concealed tiles (including winning tile)
    pub melds: Vec<MeldType>,    // exposed melds
    pub winning_tile: Tile,
    pub is_ron: bool,
    pub ding_que: Option<Suit>,
    pub is_after_kan: bool,      // for 杠上花
    pub is_kan_discard: bool,    // for 杠上炮
    pub is_chankan: bool,        // for 抢杠
    pub is_haidi: bool,          // last tile
    pub is_tianhu: bool,
    pub is_dihu: bool,
    pub exclude_gen_tile: Option<Tile>, // for chankan gen exclusion
    pub fan_config: FanConfig,
}

/// Result of fan calculation
#[derive(Debug, Clone, Default)]
pub struct FanResult {
    pub fan: u8,
    pub pinghu: bool,
    pub tsumo: bool,
    pub menqing: bool,
    pub qidui: bool,
    pub toitoi: bool,
    pub jingoudiao: bool,
    pub qingyise: bool,
    pub daiyaojiu: bool,
    pub duanyaojiu: bool,
    pub yitiaolong: bool,
    pub jiaxinwu: bool,
    pub gen_count: u8,
    pub gangshanghua: bool,
    pub gangshangpao: bool,
    pub chankan: bool,
    pub haidi: bool,
    pub tianhu_dihu: bool,
}

/// A hand division into mentsu (triplets/sequences) and jantai (pair)
#[derive(Debug, Clone)]
struct Division {
    pair_tile: Tile,
    kotsu: Vec<Tile>,   // triplet tiles
    shuntsu: Vec<Tile>, // sequence start tiles
}

/// Main agari calculation entry point
pub fn calc_fan(ctx: &WinContext) -> Option<FanResult> {
    // Cannot agari if ding que suit tiles remain in hand or melds
    if !ding_que::validate_win(&ctx.tehai, &ctx.melds, ctx.ding_que) {
        return None;
    }

    let mut result = FanResult::default();
    let mut best_fan = 0u8;

    // Base fans always applied
    result.pinghu = true; // +1 base

    if !ctx.is_ron {
        result.tsumo = true;
    }

    // Menqing: no open melds (ankan doesn't break menqing)
    let is_menqing = ctx.melds.iter().all(|m| !m.is_open());
    if is_menqing && ctx.fan_config.menqing {
        result.menqing = true;
    }

    // Gen count (四归一): tiles appearing 4 times
    let gen_count = calc_gen_count(&ctx.tehai, &ctx.melds, ctx.exclude_gen_tile);
    result.gen_count = gen_count;

    // Build full 14-tile hand for division
    let total_in_hand = ctx.tehai.iter().sum::<u8>() as usize;
    let expected = (4 - ctx.melds.len()) * 3 + 2;

    if total_in_hand != expected {
        return None;
    }

    // Try all divisions and pick the one with max fan
    let divisions = find_divisions(&ctx.tehai, ctx.melds.len());

    for div in &divisions {
        let div_fan = calc_division_fan(ctx, div, gen_count);
        if div_fan > best_fan {
            best_fan = div_fan;
            update_result_from_division(&mut result, ctx, div);
        }
    }

    // Check seven pairs (chitoi)
    if ctx.melds.is_empty() && total_in_hand == 14 {
        if is_valid_chitoi(&ctx.tehai) {
            let chitoi_fan = calc_chitoi_fan(ctx, gen_count);
            if chitoi_fan > best_fan {
                result.qidui = true;
                result.toitoi = false;
                result.jingoudiao = false;
                result.yitiaolong = false;
                result.jiaxinwu = false;
                update_chitoi_result(&mut result, ctx);
            }
        }
    }

    if divisions.is_empty() && !result.qidui {
        return None;
    }

    // Compose final fan
    let mut fan = 1u8; // pinghu base
    if result.tsumo { fan += 1; }
    if result.menqing { fan += 1; }
    if result.qidui { fan += 2; }
    if result.toitoi { fan += 1; }
    if result.jingoudiao { fan += 1; }
    if result.qingyise { fan += 2; }
    if result.daiyaojiu { fan += 3; }
    if result.duanyaojiu { fan += 1; }
    if result.yitiaolong { fan += 1; }
    if result.jiaxinwu { fan += 1; }
    fan += result.gen_count;

    // Kan-related fans
    if ctx.is_after_kan && !ctx.is_ron {
        result.gangshanghua = true;
        fan += 1;
    }
    if ctx.is_kan_discard && ctx.is_ron && !ctx.is_chankan {
        result.gangshangpao = true;
        fan += 1;
    }
    if ctx.is_chankan && ctx.is_ron {
        result.chankan = true;
        fan += 1;
    }

    // Haidi
    if ctx.is_haidi && ctx.fan_config.haidi {
        result.haidi = true;
        fan += 1;
    }

    // Tianhu/Dihu: set to MAX_FAN (6)
    if (ctx.is_tianhu || ctx.is_dihu) && ctx.fan_config.tianhu_dihu {
        result.tianhu_dihu = true;
        fan = MAX_FAN;
    }

    result.fan = fan.min(MAX_FAN);
    Some(result)
}

fn calc_division_fan(ctx: &WinContext, div: &Division, gen_count: u8) -> u8 {
    let mut fan = 1u8 + gen_count; // pinghu base + gen

    if !ctx.is_ron { fan += 1; } // tsumo
    if ctx.melds.iter().all(|m| !m.is_open()) && ctx.fan_config.menqing { fan += 1; }

    // Toitoi: all triplets (no sequences); pon/kan melds are all kotsu-type
    let all_kotsu = div.shuntsu.is_empty()
        && div.kotsu.len() + ctx.melds.len() == 4;
    if all_kotsu {
        fan += 1;
    }

    // Jingoudiao: 4 fuuro + single tile
    let total_hand = ctx.tehai.iter().sum::<u8>();
    if ctx.melds.len() == 4 && total_hand == 2 {
        fan += 1; // jingoudiao coexists with toitoi
    }

    // Qingyise
    if check_qingyise(&ctx.tehai, &ctx.melds) {
        fan += 2;
    }

    // Yitiaolong
    if ctx.fan_config.yitiaolong && check_yitiaolong(div) {
        fan += 1;
    }

    // Jiaxinwu
    if ctx.fan_config.jiaxinwu && check_jiaxinwu(ctx, div) {
        fan += 1;
    }

    // Daiyaojiu vs Duanyaojiu (mutually exclusive)
    if ctx.fan_config.daiyaojiu && check_daiyaojiu(div, &ctx.melds) {
        fan += 3;
    } else if ctx.fan_config.duanyaojiu && check_duanyaojiu(div, &ctx.melds) {
        fan += 1;
    }

    fan.min(MAX_FAN)
}

fn calc_chitoi_fan(ctx: &WinContext, gen_count: u8) -> u8 {
    let mut fan = 1u8 + 2 + gen_count; // pinghu + qidui + gen
    if !ctx.is_ron { fan += 1; }
    if ctx.fan_config.menqing { fan += 1; } // chitoi is always menqing

    if check_qingyise(&ctx.tehai, &ctx.melds) { fan += 2; }

    // Daiyaojiu/duanyaojiu for chitoi
    if ctx.fan_config.daiyaojiu && check_chitoi_daiyaojiu(&ctx.tehai) {
        fan += 3;
    } else if ctx.fan_config.duanyaojiu && check_chitoi_duanyaojiu(&ctx.tehai) {
        fan += 1;
    }

    fan.min(MAX_FAN)
}

fn update_result_from_division(result: &mut FanResult, ctx: &WinContext, div: &Division) {
    result.qidui = false;
    let all_kotsu = div.shuntsu.is_empty();
    result.toitoi = all_kotsu && div.kotsu.len() + ctx.melds.len() == 4;

    let total_hand = ctx.tehai.iter().sum::<u8>();
    result.jingoudiao = ctx.melds.len() == 4 && total_hand == 2;

    result.qingyise = check_qingyise(&ctx.tehai, &ctx.melds);
    result.yitiaolong = ctx.fan_config.yitiaolong && check_yitiaolong(div);
    result.jiaxinwu = ctx.fan_config.jiaxinwu && check_jiaxinwu(ctx, div);
    result.daiyaojiu = ctx.fan_config.daiyaojiu && check_daiyaojiu(div, &ctx.melds);
    result.duanyaojiu = !result.daiyaojiu && ctx.fan_config.duanyaojiu && check_duanyaojiu(div, &ctx.melds);
}

fn update_chitoi_result(result: &mut FanResult, ctx: &WinContext) {
    result.qingyise = check_qingyise(&ctx.tehai, &ctx.melds);
    result.daiyaojiu = ctx.fan_config.daiyaojiu && check_chitoi_daiyaojiu(&ctx.tehai);
    result.duanyaojiu = !result.daiyaojiu && ctx.fan_config.duanyaojiu && check_chitoi_duanyaojiu(&ctx.tehai);
}

pub fn calc_gen_count(hand: &HandCounts, melds: &[MeldType], exclude: Option<Tile>) -> u8 {
    let mut full = [0u8; NUM_TILE_TYPES];
    for (i, &c) in hand.iter().enumerate() {
        full[i] += c;
    }
    for m in melds {
        let t = m.tile() as usize;
        full[t] += m.tile_count();
    }
    let mut count = 0u8;
    for (i, &c) in full.iter().enumerate() {
        if c >= 4 {
            if let Some(ex) = exclude {
                if i == ex as usize { continue; }
            }
            count += 1;
        }
    }
    count
}

fn check_qingyise(hand: &HandCounts, melds: &[MeldType]) -> bool {
    let mut suit: Option<Suit> = None;
    for (i, &c) in hand.iter().enumerate() {
        if c > 0 {
            let s = Suit::from_tile(i as u8);
            match suit {
                None => suit = Some(s),
                Some(prev) => if prev != s { return false; },
            }
        }
    }
    for m in melds {
        let s = Suit::from_tile(m.tile());
        match suit {
            None => suit = Some(s),
            Some(prev) => if prev != s { return false; },
        }
    }
    suit.is_some()
}

fn check_yitiaolong(div: &Division) -> bool {
    // Need 123, 456, 789 of same suit in shuntsu
    for &suit in &[Suit::Man, Suit::Pin, Suit::Sou] {
        let base = suit.start();
        let has_123 = div.shuntsu.contains(&(base as Tile));
        let has_456 = div.shuntsu.contains(&((base + 3) as Tile));
        let has_789 = div.shuntsu.contains(&((base + 6) as Tile));
        if has_123 && has_456 && has_789 {
            return true;
        }
    }
    false
}

fn check_jiaxinwu(ctx: &WinContext, div: &Division) -> bool {
    let wt = ctx.winning_tile;
    if Suit::rank(wt) != 5 { return false; }

    let suit_start = Suit::from_tile(wt).start();
    let seq_start = (suit_start + 3) as Tile; // 456 starts at rank 4
    if !div.shuntsu.contains(&seq_start) { return false; }

    // Count how many copies of tile5 are consumed by groups OTHER than
    // the first 456 shuntsu in this division.
    let tile5 = wt as usize;
    let mut other_usage: u8 = 0;

    // Pair
    if div.pair_tile == wt { other_usage += 2; }

    // Kotsu (triplets)
    for &k in &div.kotsu {
        if k == wt { other_usage += 3; }
    }

    // Shuntsu: tile5 appears in sequences starting at rank 3,4,5
    // (i.e. 345, 456, 567) of the same suit.
    let mut found_target_456 = false;
    for &s in &div.shuntsu {
        if s == seq_start && !found_target_456 {
            found_target_456 = true;
            continue; // skip the first 456 — that's the one we're checking
        }
        if Suit::from_tile(s).start() == suit_start {
            let s_rank = Suit::rank(s);
            // sequence [s_rank, s_rank+1, s_rank+2] contains rank 5
            // when s_rank <= 5 && s_rank+2 >= 5, i.e. s_rank in {3,4,5}
            if s_rank >= 3 && s_rank <= 5 {
                other_usage += 1;
            }
        }
    }

    // If tehai[5] == other_usage + 1, the only remaining copy of tile5
    // is the one used by the 456 shuntsu — confirmed kanchan wait.
    ctx.tehai[tile5] == other_usage + 1
}

fn check_daiyaojiu(div: &Division, melds: &[MeldType]) -> bool {
    // Every group must contain a terminal (1 or 9)
    if !is_terminal(div.pair_tile) {
        return false;
    }
    // Kotsu from hand
    for &t in &div.kotsu {
        if !is_terminal(t) { return false; }
    }
    // Shuntsu: must contain 1 or 9 → only 1-2-3 or 7-8-9
    for &t in &div.shuntsu {
        let rank = Suit::rank(t);
        if rank != 1 && rank != 7 { return false; }
    }
    // Melds
    for m in melds {
        if !is_terminal(m.tile()) { return false; }
    }
    true
}

fn check_duanyaojiu(div: &Division, melds: &[MeldType]) -> bool {
    // All tiles must be 2-8
    let is_inner = |t: Tile| -> bool {
        let r = Suit::rank(t);
        r >= 2 && r <= 8
    };
    if !is_inner(div.pair_tile) { return false; }
    for &t in &div.kotsu {
        if !is_inner(t) { return false; }
    }
    for &t in &div.shuntsu {
        let r = Suit::rank(t);
        // sequence start rank must be 2-6 (so end is 4-8)
        if r < 2 || r > 6 { return false; }
    }
    for m in melds {
        if !is_inner(m.tile()) { return false; }
    }
    true
}

fn check_chitoi_daiyaojiu(hand: &HandCounts) -> bool {
    for (i, &c) in hand.iter().enumerate() {
        if c > 0 && !is_terminal(i as u8) { return false; }
    }
    true
}

fn check_chitoi_duanyaojiu(hand: &HandCounts) -> bool {
    for (i, &c) in hand.iter().enumerate() {
        if c > 0 {
            let r = Suit::rank(i as u8);
            if r == 1 || r == 9 { return false; }
        }
    }
    true
}

fn is_valid_chitoi(hand: &HandCounts) -> bool {
    let mut pairs = 0;
    for &c in hand.iter() {
        if c % 2 != 0 { return false; }
        pairs += c / 2;
    }
    pairs == 7
}

/// Find all valid standard-form divisions of a hand
fn find_divisions(hand: &HandCounts, num_melds: usize) -> Vec<Division> {
    use std::collections::HashSet;

    let target_mentsu = 4 - num_melds;
    let mut results = Vec::new();
    let mut seen: HashSet<(Tile, Vec<Tile>, Vec<Tile>)> = HashSet::new();
    let mut h = *hand;

    for pair_tile in 0..NUM_TILE_TYPES {
        if h[pair_tile] < 2 { continue; }
        h[pair_tile] -= 2;

        let mut all_divs = Vec::new();
        find_all_mentsu(&mut h, 0, target_mentsu, &mut Vec::new(), &mut Vec::new(), &mut all_divs);
        for (k, s) in all_divs {
            let key = (pair_tile as Tile, k.clone(), s.clone());
            if seen.insert(key) {
                results.push(Division {
                    pair_tile: pair_tile as Tile,
                    kotsu: k,
                    shuntsu: s,
                });
            }
        }

        h[pair_tile] += 2;
    }

    results
}

fn find_all_mentsu(
    hand: &mut HandCounts,
    start: usize,
    target: usize,
    kotsu: &mut Vec<Tile>,
    shuntsu: &mut Vec<Tile>,
    results: &mut Vec<(Vec<Tile>, Vec<Tile>)>,
) {
    if kotsu.len() + shuntsu.len() == target {
        if hand.iter().all(|&c| c == 0) {
            results.push((kotsu.clone(), shuntsu.clone()));
        }
        return;
    }

    for i in start..NUM_TILE_TYPES {
        if hand[i] == 0 { continue; }

        if hand[i] >= 3 {
            hand[i] -= 3;
            kotsu.push(i as Tile);
            find_all_mentsu(hand, i, target, kotsu, shuntsu, results);
            kotsu.pop();
            hand[i] += 3;
        }

        let suit_of_i = i / TILES_PER_SUIT;
        if i + 2 < NUM_TILE_TYPES && (i + 2) / TILES_PER_SUIT == suit_of_i {
            if hand[i] >= 1 && hand[i + 1] >= 1 && hand[i + 2] >= 1 {
                hand[i] -= 1;
                hand[i + 1] -= 1;
                hand[i + 2] -= 1;
                shuntsu.push(i as Tile);
                find_all_mentsu(hand, i, target, kotsu, shuntsu, results);
                shuntsu.pop();
                hand[i] += 1;
                hand[i + 1] += 1;
                hand[i + 2] += 1;
            }
        }

        break;
    }
}
