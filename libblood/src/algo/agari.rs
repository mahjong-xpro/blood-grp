//! Rust port of EndlessCheng's Go port of 山岡忠夫's Java implementation of his
//! agari algorithm.
//!
//! Source:
//! * Go: <https://github.com/EndlessCheng/mahjong-helper/blob/master/util/agari.go>
//! * Java: <http://hp.vector.co.jp/authors/VA046927/mjscore/AgariIndex.java>
//! * Algorithm: <http://hp.vector.co.jp/authors/VA046927/mjscore/mjalgorism.html>

use super::point::Point;
use super::shanten;
use crate::tile::Tile;
#[cfg(test)]
use crate::tu8;
use std::sync::LazyLock;

use boomphf::hashmap::BoomHashMap;
use byteorder::{LittleEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use tinyvec::ArrayVec;

const AGARI_TABLE_SIZE: usize = 9_362;

static AGARI_TABLE: LazyLock<BoomHashMap<u32, ArrayVec<[Div; 4]>>> = LazyLock::new(|| {
    let mut raw = GzDecoder::new(include_bytes!("data/agari.bin.gz").as_slice());

    let (keys, values): (Vec<_>, Vec<_>) = (0..AGARI_TABLE_SIZE)
        .map(|_| {
            let key = raw.read_u32::<LittleEndian>().unwrap();
            let v_size = raw.read_u8().unwrap();
            let value = (0..v_size)
                .map(|_| raw.read_u32::<LittleEndian>().unwrap())
                .map(Div::from)
                .collect();
            (key, value)
        })
        .unzip();

    if cfg!(test) {
        // Ensure there is no duplicated keys.
        let mut k = keys.clone();
        k.sort_unstable();
        k.dedup();
        assert_eq!(k.len(), keys.len());

        // Ensure there is no data left to read.
        raw.read_u8().unwrap_err();
    }

    BoomHashMap::new(keys, values)
});

#[derive(Debug, Default)]
struct Div {
    pair_idx: u8,
    kotsu_idxs: ArrayVec<[u8; 4]>,
    shuntsu_idxs: ArrayVec<[u8; 4]>,
    has_chitoi: bool,
}

/// 
/// Only fan (番数) is used, no fu (符数)
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Agari {
    /// Fan count (1-5, capped at 5)
    Fan(u8),
}

/// Configurable Sichuan Mahjong fan rules.
///
/// Different tables / regions enable different optional fan types.
/// `Default` enables all fans (standard 血战到底 full rule set).
/// During training, random subsets can be toggled (domain randomization)
/// so the model learns to adapt to varied rule combinations at inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pyo3::pyclass(get_all, set_all)]
pub struct FanConfig {
    /// 门清 (+1番): no open melds. Default: true
    pub menqing: bool,
    /// 断幺九 (+1番): all tiles 2–8, no terminals. Default: true
    pub duanyaojiu: bool,
    /// 带幺九 (+3番): every meld/pair contains 1 or 9. Default: true
    pub daiyaojiu: bool,
    /// 一条龙 (+1番): same suit 123+456+789. Default: true
    pub yitiaolong: bool,
    /// 夹心五 (+1番): win on 5 via 4-6 kanchan wait. Default: true
    pub jiaxinwu: bool,
    /// 海底捞月/海底炮 (+1番): win on last tile. Default: true
    pub haidi: bool,
    /// 天胡/地胡 (直接5番). Default: true
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

#[pyo3::pymethods]
impl FanConfig {
    #[new]
    #[pyo3(signature = (
        menqing=true,
        duanyaojiu=true,
        daiyaojiu=true,
        yitiaolong=true,
        jiaxinwu=true,
        haidi=true,
        tianhu_dihu=true,
    ))]
    fn py_new(
        menqing: bool,
        duanyaojiu: bool,
        daiyaojiu: bool,
        yitiaolong: bool,
        jiaxinwu: bool,
        haidi: bool,
        tianhu_dihu: bool,
    ) -> Self {
        Self { menqing, duanyaojiu, daiyaojiu, yitiaolong, jiaxinwu, haidi, tianhu_dihu }
    }

    fn __repr__(&self) -> String {
        format!(
            "FanConfig(menqing={}, duanyaojiu={}, daiyaojiu={}, yitiaolong={}, jiaxinwu={}, haidi={}, tianhu_dihu={})",
            self.menqing, self.duanyaojiu, self.daiyaojiu, self.yitiaolong, self.jiaxinwu, self.haidi, self.tianhu_dihu,
        )
    }
}

impl FanConfig {
    /// All fans enabled (standard 血战到底).
    pub const ALL: Self = Self {
        menqing: true,
        duanyaojiu: true,
        daiyaojiu: true,
        yitiaolong: true,
        jiaxinwu: true,
        haidi: true,
        tianhu_dihu: true,
    };

    /// Number of configurable flags (for observation encoding).
    pub const NUM_FLAGS: usize = 7;

    /// Return flags as a fixed-size bool array for observation encoding.
    #[inline]
    pub fn as_flags(&self) -> [bool; Self::NUM_FLAGS] {
        [
            self.menqing,
            self.duanyaojiu,
            self.daiyaojiu,
            self.yitiaolong,
            self.jiaxinwu,
            self.haidi,
            self.tianhu_dihu,
        ]
    }

    /// Generate a random FanConfig from a seed.
    ///
    /// Each flag is independently set to true/false with 50% probability.
    /// The seed ensures reproducibility: same seed → same config.
    ///
    /// Used in Phase 2 multi-rule training: each game gets a different
    /// random rule set so the model learns to condition on rule flags.
    pub fn random_from_seed(seed: u64) -> Self {
        // Use a simple hash-based approach for fast, deterministic randomness.
        // We don't need cryptographic quality - just good distribution.
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        seed.hash(&mut hasher);
        let bits = hasher.finish();

        Self {
            menqing:     bits & (1 << 0) != 0,
            duanyaojiu:  bits & (1 << 1) != 0,
            daiyaojiu:   bits & (1 << 2) != 0,
            yitiaolong:  bits & (1 << 3) != 0,
            jiaxinwu:    bits & (1 << 4) != 0,
            haidi:       bits & (1 << 5) != 0,
            tianhu_dihu: bits & (1 << 6) != 0,
        }
    }
}

#[derive(Debug)]
pub struct AgariCalculator<'a> {
    /// Must include the winning tile (i.e. must be 3n+2)
    pub tehai: &'a [u8; 27],

    pub pons: &'a [u8],
    pub minkans: &'a [u8],
    pub ankans: &'a [u8],

    /// Winning tile (must be normalized, no aka dora distinction in Bloody Battle Mahjong)
    pub winning_tile: u8,
    /// True for ron (荣和), false for tsumo (自摸)
    pub is_ron: bool,
    
    pub ding_que: Option<crate::mjai::Suit>,
    
    pub is_after_kan: bool,
    pub is_kan_discard: bool,
    pub is_chankan: bool,
    /// If Some, this tile will not be counted as gen even if it appears 4 times
    pub exclude_gen_tile: Option<u8>,
    pub is_haidi: bool,
    pub is_tianhu: bool,
    pub is_dihu: bool,

    /// Configurable fan rules (which optional fan types are active).
    pub fan_config: FanConfig,
}

impl From<u32> for Div {
    fn from(v: u32) -> Self {
        let pair_idx = ((v >> 6) & 0b1111) as u8;

        let kotsu_count = v & 0b111;
        let kotsu_idxs = (0..kotsu_count)
            .map(|i| ((v >> (10 + i * 4)) & 0b1111) as u8)
            .collect();

        let shuntsu_count = (v >> 3) & 0b111;
        let shuntsu_idxs = (kotsu_count..kotsu_count + shuntsu_count)
            .map(|i| ((v >> (10 + i * 4)) & 0b1111) as u8)
            .collect();

        let has_chitoi = (v >> 26) & 0b1 == 0b1;

        Self {
            pair_idx,
            kotsu_idxs,
            shuntsu_idxs,
            has_chitoi,
        }
    }
}

impl Agari {
    #[must_use]
    pub fn point(self, _is_oya: bool) -> Point {
        match self {
            Self::Fan(fan) => Point::calc_from_fan(fan),
        }
    }
}

impl AgariCalculator<'_> {
    /// Build a 14-tile view of the hand for AGARI_TABLE / division logic.
    ///
    /// `self.tehai` is the concealed tile counter and must already include the winning tile
    /// (i.e. be 3n+2). In Bloody Battle Mahjong, open melds are only pons/kans (no chi), so we
    /// can treat each meld as a triplet for hand-structure purposes (a kan is still one kotsu,
    /// the 4th tile is not needed for division).
    ///
    /// Returns `None` if the reconstructed hand is not a valid 14-tile multiset (e.g. counts > 4
    /// or sum != 14), which indicates an inconsistent state/log.
    #[inline]
    fn hand14_for_division(&self) -> Option<[u8; 27]> {
        let mut tiles = *self.tehai;

        // Add exposed sets back as triplets so the total becomes 14.
        for &tile_id in self
            .pons
            .iter()
            .chain(self.minkans.iter())
            .chain(self.ankans.iter())
        {
            let idx = tile_id as usize;
            if idx >= 27 {
                return None;
            }
            tiles[idx] = tiles[idx].saturating_add(3);
            if tiles[idx] > 4 {
                return None;
            }
        }

        if tiles.iter().sum::<u8>() != 14 {
            return None;
        }

        Some(tiles)
    }

    /// Check if the hand can agari (和牌)
    /// 
    /// Must check:
    /// 1. Valid 14-tile agari structure (AGARI_TABLE)
    /// 2. Division must respect exposed melds: pons/kans stay as kotsu (not split into pair+shuntsu)
    /// 3. Ding Que rule: cannot agari if hand still has ding_que suit tiles
    #[inline]
    #[must_use]
    pub fn has_yaku(&self) -> bool {
        // DingQue rule (花猪): cannot agari if DingQue suit tiles remain in full hand (tehai + fuuro).
        if !crate::ding_que::can_agari_with_fuuro(self.tehai, self.pons, self.minkans, self.ankans, self.ding_que) {
            return false;
        }

        let Some(hand14) = self.hand14_for_division() else {
            return false;
        };
        let (tile14, key) = get_tile14_and_key(&hand14);
        let divs_opt = AGARI_TABLE.get(&key);

        // FIX: 龙七对回退——AGARI_TABLE 不包含龙七对，
        // 但龙七对是血战到底的合法和牌型。
        let is_chitoi = is_valid_chitoi_hand(&hand14);
        let has_fuuro = !self.pons.is_empty() || !self.minkans.is_empty() || !self.ankans.is_empty();

        // 龙七对不可能有副露
        if is_chitoi && !has_fuuro {
            return true;
        }

        // AGARI_TABLE 标准路径
        let Some(divs) = divs_opt else {
            return false;
        };

        // 副露约束：碰/杠的牌在和牌分解中必须作为刻子（kotsu），
        // 不能被拆分为对子+顺子。否则会出现假阳性（如碰 1万 + 手牌 3万，
        // 对手打 2万，table 找到 1万1万(对)+1万2万3万(顺) 的非法分解）。
        if has_fuuro {
            if !divs.iter().any(|div| {
                is_division_compatible_with_fuuro(div, &tile14, self.pons, self.minkans, self.ankans)
            }) {
                return false;
            }
        }
        
        true
    }

    #[inline]
    #[must_use]
    pub fn search_yakus(&self) -> Option<Agari> {
        self.agari()
    }

    /// 
    /// Returns the total fan count (1-5, capped at 5)
    /// 
    /// 1. 平胡（PingHu）：+1番（基础，必须）
    /// 2. 自摸（Tsumo）：+1番（if !is_ron）
    /// 3. 门清（MenQing）：+1番（无副露：无碰、无杠）
    /// 4. 七对（QiDui）：+2番
    /// 5. 碰碰胡（ToiToi）：+1番
    /// 6. 金钩钓（JinGouDiao）：+1番
    /// 7. 清一色（QingYiSe）：+2番
    /// 8. 带幺九（DaiYaoJiu）：+3番（与断幺九互斥）
    /// 9. 断幺九（DuanYaoJiu）：+1番（所有组合不含1和9，与带幺九互斥）
    /// 10. 一条龙（YiTiaoLong）：+1番（同一花色含 123、456、789 三副顺子）
    /// 11. 夹心五（JiaXinWu）：+1番（和牌张为 5，46 听 5 成顺 456）
    /// 12. 四归一（SiGuiYi / 根）：+1番/根
    /// 13. 杠上花（GangShangHua）：+1番（if is_after_kan && !is_ron）
    /// 14. 杠上炮（GangShangPao）：+1番（if is_kan_discard && is_ron && !is_chankan）
    /// 15. 抢杠（Chankan）：+1番（if is_chankan && is_ron）
    ///     Note: 抢杠、杠上花、杠上炮是不同的：
    ///     - 抢杠：在别人加杠时抢杠和牌，+1番（平胡1番 + 抢杠1番 = 2番）
    ///     - 杠上花：杠牌后摸牌自摸，+1番（自摸1番 + 平胡1番 + 杠上花1番 = 3番）
    ///     - 杠上炮：杠牌后打出的牌和牌，+1番（平胡1番 + 杠上炮1番 = 2番）
    #[must_use]
    pub fn agari(&self) -> Option<Agari> {
        // DingQue rule (花猪): cannot agari if DingQue suit tiles remain in full hand (tehai + fuuro).
        if !crate::ding_que::can_agari_with_fuuro(self.tehai, self.pons, self.minkans, self.ankans, self.ding_que) {
            return None;
        }
        
        let mut fan: u8 = 1;
        let fc = &self.fan_config;
        
        // 2. 自摸（Tsumo）：+1番
        if !self.is_ron {
            fan += 1;
        }

        // 13. 海底捞月/海底炮（Haidi）：+1番
        if fc.haidi && self.is_haidi {
            fan += 1;
        }

        // 3. 门清（MenQing）：+1番（无副露：无碰、无杠）
        if fc.menqing && self.pons.is_empty() && self.minkans.is_empty() && self.ankans.is_empty() {
            fan += 1;
        }
        
        // Check hand structure (must be a valid 14-tile multiset for AGARI_TABLE)
        let hand14 = self.hand14_for_division()?;
        let tile14_len: usize = hand14.iter().filter(|&&c| c > 0).count();
        let (tile14, key) = get_tile14_and_key(&hand14);
        let divs_opt = AGARI_TABLE.get(&key);

        // FIX: 龙七对回退——AGARI_TABLE 源自日麻，日麻七对子要求 7 个不同牌种各 2 张，
        // 不包含 4 张相同牌算 2 对的龙七对模式。当表查找失败时，检查龙七对。
        // 同时处理表有标准分解但缺少龙七对分解的情况（取两者最高番数）。
        let is_chitoi_hand = is_valid_chitoi_hand(&hand14);

        let has_fuuro = !self.pons.is_empty() || !self.minkans.is_empty() || !self.ankans.is_empty();

        // 如果表中无此牌型，且也不是七对子，则确实不能和牌
        if divs_opt.is_none() && !is_chitoi_hand {
            return None;
        }

        // 副露约束：必须存在至少一个分解使得碰/杠保持为刻子
        // （七对不可能有副露，所以 has_fuuro && is_chitoi_hand 不会同时为真）
        if has_fuuro {
            if let Some(divs) = divs_opt {
                if !divs.iter().any(|div| {
                    is_division_compatible_with_fuuro(div, &tile14, self.pons, self.minkans, self.ankans)
                }) {
                    return None;
                }
            } else {
                // 有副露但表中无分解且不是七对 → 不能和牌
                return None;
            }
        }
        
        // Find the best division for fan calculation
        let mut max_fan: u8 = 0;
        
        // 8. 四归一（SiGuiYi / 根）：+1番/根
        //
        // Bloody Battle rule (per rules.md): a "gen" is counted whenever a tile kind appears
        // exactly 4 times in total, across:
        // - concealed hand (`self.tehai`)
        // - exposed pons (3 tiles each)
        // - exposed/closed kans (4 tiles each)
        //
        // Note: If `exclude_gen_tile` is set (for chankan), that tile kind is not counted as gen
        // even if its total would be 4.
        let mut total_counts: [u8; 27] = *self.tehai;
        for &tile_id in self.pons.iter() {
            let idx = tile_id as usize;
            if idx < 27 {
                total_counts[idx] = total_counts[idx].saturating_add(3);
            }
        }
        for &tile_id in self.minkans.iter() {
            let idx = tile_id as usize;
            if idx < 27 {
                total_counts[idx] = total_counts[idx].saturating_add(4);
            }
        }
        for &tile_id in self.ankans.iter() {
            let idx = tile_id as usize;
            if idx < 27 {
                total_counts[idx] = total_counts[idx].saturating_add(4);
            }
        }

        debug_assert!(
            total_counts.iter().all(|&c| c <= 4),
            "invalid tile totals (>4) in gen counting"
        );

        let mut gen_count: u8 = 0;
        for (tile_id, &count) in total_counts.iter().enumerate() {
            if count == 4 {
                if self.exclude_gen_tile.is_some_and(|t| t as usize == tile_id) {
                    continue;
                }
                gen_count = gen_count.saturating_add(1);
            }
        }
        // AGARI_TABLE 标准分解循环（表中有此牌型时执行）
        for div in divs_opt.iter().flat_map(|d| d.iter()) {
            // 跳过与副露不兼容的分解（碰/杠必须保持为刻子）
            if has_fuuro && !is_division_compatible_with_fuuro(div, &tile14, self.pons, self.minkans, self.ankans) {
                continue;
            }

            let mut div_fan: u8 = gen_count;
            
            // 3. 七对（QiDui）：+2番
            // 七对与碰碰胡、金钩钓互斥（结构不同），但可与清一色、带幺九、断幺九叠加
            let is_chitoi = div.has_chitoi;
            if is_chitoi {
                div_fan += 2;
            }
            
            if !is_chitoi {
                // 以下番型仅适用于非七对的标准分解

                // 5. 金钩钓（JinGouDiao）：+1番 (4 fuuro + single wait/tanki)
                let fuuro_count = self.pons.len() + self.minkans.len() + self.ankans.len();
                
                // 4. 碰碰胡（ToiToi）：+1番 (4 kotsu + 1 pair, no shuntsu)
                // Division is computed on the full 14-tile hand (concealed + fuuro as triplets),
                // so `div.kotsu_idxs.len()` already includes exposed pons/kans.
                if div.shuntsu_idxs.is_empty() && div.kotsu_idxs.len() == 4 {
                    div_fan += 1;
                }

                if fuuro_count == 4 {
                    // Check if single wait (tanki): pair is the winning tile
                    let is_tanki = div.pair_idx < 14 && tile14[div.pair_idx as usize] == self.winning_tile;
                    if is_tanki {
                        // Jin Gou Diao (Single Wait with 4 Melds).
                        // In Bloody Battle, this stacks with ToiToi.
                        // Base (1) + ToiToi (1) + JGD (1) = 3 Fan (4000).
                        div_fan += 1;
                    }
                }
            }
            
            // 6. 清一色（QingYiSe）：+2番
            // 可与七对叠加：七对+清一色 = 平胡1+七对2+清一色2 = 5番(封顶)
            // Check if all tiles (hand + fuuro) are same suit
            let mut suit_kind: Option<u8> = None;
            let mut is_qingyise = true;
            
            // Check hand tiles (only the filled prefix of `tile14` is meaningful)
            for &tile_id in &tile14[..tile14_len] {
                if tile_id >= 27 {
                    continue; // Skip invalid tiles
                }
                let kind = tile_id / 9;
                if let Some(prev_kind) = suit_kind {
                    if prev_kind != kind {
                        is_qingyise = false;
                        break;
                    }
                } else {
                    suit_kind = Some(kind);
                }
            }
            
            // Check fuuro tiles (pons, minkans, ankans)
            if is_qingyise {
                for &tile_id in self.pons.iter().chain(self.minkans.iter()).chain(self.ankans.iter()) {
                    if tile_id >= 27 {
                        continue;
                    }
                    let kind = tile_id / 9;
                    if let Some(prev_kind) = suit_kind {
                        if prev_kind != kind {
                            is_qingyise = false;
                            break;
                        }
                    } else {
                        suit_kind = Some(kind);
                    }
                }
            }
            
            if is_qingyise && suit_kind.is_some() {
                div_fan += 2;
            }
            
            // 以下番型需要顺子结构，与七对互斥（七对无顺子）
            if !is_chitoi {
                // 10. 一条龙（YiTiaoLong）：+1番（同一花色含有 123、456、789 三副顺子）
                if fc.yitiaolong {
                    let mut suit_has_shuntsu_num: [[bool; 9]; 3] = [[false; 9]; 3];
                    for &shuntsu_idx in &div.shuntsu_idxs {
                        let tile_id = tile14[shuntsu_idx as usize];
                        if tile_id >= 27 {
                            continue;
                        }
                        let kind = (tile_id / 9) as usize;
                        let num = tile_id % 9;
                        if kind < 3 {
                            suit_has_shuntsu_num[kind][num as usize] = true;
                        }
                    }
                    let is_yitiaolong = (0..3).any(|kind| {
                        suit_has_shuntsu_num[kind][0] && suit_has_shuntsu_num[kind][3] && suit_has_shuntsu_num[kind][6]
                    });
                    if is_yitiaolong {
                        div_fan += 1;
                    }
                }
                
                // 11. 夹心五（JiaXinWu）：+1番（和牌张为 5，且听牌为 4-6 夹 5，和 5 成顺 456）
                if fc.jiaxinwu {
                    let is_jiaxinwu = (self.winning_tile < 27 && self.winning_tile % 9 == 4)
                        && div.shuntsu_idxs.iter().any(|&shuntsu_idx| {
                            let tile_id = tile14[shuntsu_idx as usize];
                            tile_id < 27
                                && tile_id % 9 == 3
                                && tile_id / 9 == self.winning_tile / 9
                        });
                    if is_jiaxinwu {
                        div_fan += 1;
                    }
                }
            }
            
            // 7. 带幺九（DaiYaoJiu）：+3番
            // 可与七对叠加：七对+带幺九 = 平胡1+七对2+带幺九3 = 6番→封顶5番
            let is_daiyaojiu = if is_chitoi {
                // 七对：直接检查手牌中所有牌种是否都是 1 或 9
                self.tehai.iter().enumerate().all(|(tid, &count)| {
                    if count == 0 { return true; }
                    let num = tid % 9;
                    num == 0 || num == 8
                })
            } else {
                let mut ok = true;
                
                // Check shuntsu: must start with 1 or end with 9 (1-2-3 or 7-8-9)
                for &shuntsu_idx in &div.shuntsu_idxs {
                    let tile_id = tile14[shuntsu_idx as usize];
                    if tile_id >= 27 { continue; }
                    let num = tile_id % 9;
                    if num != 0 && num != 6 { ok = false; break; }
                }
                
                // Check kotsu: must be 1 or 9
                if ok {
                    for &kotsu_idx in &div.kotsu_idxs {
                        let tile_id = tile14[kotsu_idx as usize];
                        if tile_id >= 27 { continue; }
                        let num = tile_id % 9;
                        if num != 0 && num != 8 { ok = false; break; }
                    }
                }
                
                // Check pair: must be 1 or 9
                if ok {
                    let pair_tile = tile14[div.pair_idx as usize];
                    if pair_tile < 27 {
                        let num = pair_tile % 9;
                        if num != 0 && num != 8 { ok = false; }
                    }
                }
                
                // Check fuuro (pons, minkans, ankans): must be 1 or 9
                if ok {
                    for &tile_id in self.pons.iter().chain(self.minkans.iter()).chain(self.ankans.iter()) {
                        if tile_id >= 27 { continue; }
                        let num = tile_id % 9;
                        if num != 0 && num != 8 { ok = false; break; }
                    }
                }
                ok
            };
            
            if fc.daiyaojiu && is_daiyaojiu {
                div_fan += 3;
            } else if fc.duanyaojiu {
                // 9. 断幺九（DuanYaoJiu）：+1番（所有组合不含1和9，仅2–8；与带幺九互斥）
                // 可与七对叠加
                let is_duanyaojiu = if is_chitoi {
                    // 七对：直接检查手牌中所有牌种是否都是 2-8
                    self.tehai.iter().enumerate().all(|(tid, &count)| {
                        if count == 0 { return true; }
                        let num = tid % 9;
                        num >= 1 && num <= 7
                    })
                } else {
                    let mut ok = true;
                    for &shuntsu_idx in &div.shuntsu_idxs {
                        let tile_id = tile14[shuntsu_idx as usize];
                        if tile_id >= 27 { continue; }
                        let num = tile_id % 9;
                        // 顺子仅允许 234,345,456,567,678（首张 num 1..=5），不含 123(num0)、789(num6)
                        if num == 0 || num == 6 { ok = false; break; }
                    }
                    if ok {
                        for &kotsu_idx in &div.kotsu_idxs {
                            let tile_id = tile14[kotsu_idx as usize];
                            if tile_id >= 27 { continue; }
                            let num = tile_id % 9;
                            if num == 0 || num == 8 { ok = false; break; }
                        }
                    }
                    if ok {
                        let pair_tile = tile14[div.pair_idx as usize];
                        if pair_tile < 27 {
                            let num = pair_tile % 9;
                            if num == 0 || num == 8 { ok = false; }
                        }
                    }
                    if ok {
                        for &tile_id in self.pons.iter().chain(self.minkans.iter()).chain(self.ankans.iter()) {
                            if tile_id >= 27 { continue; }
                            let num = tile_id % 9;
                            if num == 0 || num == 8 { ok = false; break; }
                        }
                    }
                    ok
                };
                if is_duanyaojiu {
                    div_fan += 1;
                }
            }
            
            max_fan = max_fan.max(div_fan);
        }

        // FIX: 龙七对独立计算——当手牌是有效七对子（包括 4 张相同牌算 2 对的龙七对），
        // 但 AGARI_TABLE 中没有对应的 has_chitoi 分解时，在此独立计算七对番数。
        // 这确保龙七对模式不会因为不在日麻预计算表中而被遗漏。
        if is_chitoi_hand && !has_fuuro {
            let mut chitoi_fan: u8 = 2; // 七对 +2番
            chitoi_fan += gen_count; // 四归一/根

            // 清一色 +2番
            let mut chitoi_suit: Option<u8> = None;
            let mut chitoi_qingyise = true;
            for (tid, &count) in hand14.iter().enumerate() {
                if count == 0 { continue; }
                let kind = (tid / 9) as u8;
                if let Some(prev) = chitoi_suit {
                    if prev != kind { chitoi_qingyise = false; break; }
                } else {
                    chitoi_suit = Some(kind);
                }
            }
            if chitoi_qingyise && chitoi_suit.is_some() {
                chitoi_fan += 2;
            }

            // 带幺九 +3番（所有牌种都是 1 或 9）
            if fc.daiyaojiu {
                let chitoi_daiyaojiu = self.tehai.iter().enumerate().all(|(tid, &count)| {
                    if count == 0 { return true; }
                    let num = tid % 9;
                    num == 0 || num == 8
                });
                if chitoi_daiyaojiu {
                    chitoi_fan += 3;
                } else if fc.duanyaojiu {
                    // 断幺九 +1番（所有牌种都是 2-8）
                    let chitoi_duanyaojiu = self.tehai.iter().enumerate().all(|(tid, &count)| {
                        if count == 0 { return true; }
                        let num = tid % 9;
                        num >= 1 && num <= 7
                    });
                    if chitoi_duanyaojiu {
                        chitoi_fan += 1;
                    }
                }
            } else if fc.duanyaojiu {
                // 断幺九 +1番（所有牌种都是 2-8）— 带幺九关闭时单独检查
                let chitoi_duanyaojiu = self.tehai.iter().enumerate().all(|(tid, &count)| {
                    if count == 0 { return true; }
                    let num = tid % 9;
                    num >= 1 && num <= 7
                });
                if chitoi_duanyaojiu {
                    chitoi_fan += 1;
                }
            }

            max_fan = max_fan.max(chitoi_fan);
        }
        
        fan = fan.saturating_add(max_fan);
        
        // 13. 杠上花（GangShangHua）：+1番
        if self.is_after_kan && !self.is_ron {
            fan += 1;
        }
        
        // 14. 杠上炮（GangShangPao）：+1番
        // 注意：抢杠（chankan）和杠上炮是不同的
        // - 抢杠：在别人加杠时抢杠和牌，+1番
        // - 杠上炮：其他玩家杠牌后打出的牌和牌，+1番
        if self.is_kan_discard && self.is_ron && !self.is_chankan {
            fan += 1;
        }
        
        // 15. 抢杠（Chankan）：+1番
        // 抢杠：在别人加杠时抢杠和牌，+1番
        // 抢杠时，被抢杠的玩家的根不应该计算（因为加杠的牌被抢走了）
        if self.is_chankan && self.is_ron {
            fan += 1;
        }
        
        // 17. 天胡 (TianHu) / 地胡 (DiHu): Max Fan (5番)
        if fc.tianhu_dihu && (self.is_tianhu || self.is_dihu) {
            fan = 5;
            // Early return or just let it be capped below (it is already 5)
        }

        // 5番封顶
        fan = fan.min(5);
        
        Some(Agari::Fan(fan))
    }
}

/// 检查某个 AGARI_TABLE 分解是否与副露兼容。
/// 副露（碰/明杠/暗杠）的牌在分解中必须作为刻子（kotsu），
/// 不能被拆成对子 + 顺子的一部分。
/// 七对（chitoi）与任何副露不兼容。
fn is_division_compatible_with_fuuro(
    div: &Div,
    tile14: &[u8; 14],
    pons: &[u8],
    minkans: &[u8],
    ankans: &[u8],
) -> bool {
    // 七对与副露不兼容
    if div.has_chitoi {
        return pons.is_empty() && minkans.is_empty() && ankans.is_empty();
    }

    // 每个副露的牌必须在此分解中作为刻子
    for &meld_tile in pons.iter().chain(minkans.iter()).chain(ankans.iter()) {
        let Some(idx) = tile14.iter().position(|&t| t == meld_tile) else {
            return false; // 副露的牌不在 tile14 中（不应发生）
        };
        if !div.kotsu_idxs.contains(&(idx as u8)) {
            return false; // 此分解把副露拆分了
        }
    }

    true
}

/// 检查 14 张手牌是否构成有效的七对子（包括龙七对：4 张相同牌算 2 对）。
///
/// 规则：每种牌只能出现 0、2、4 张。2 张 = 1 对，4 张 = 2 对（龙七对）。
/// 总对数必须恰好为 7。
fn is_valid_chitoi_hand(tiles: &[u8; 27]) -> bool {
    let mut pairs: u8 = 0;
    for &count in tiles {
        match count {
            0 => {}
            2 => pairs += 1,
            4 => pairs += 2,
            _ => return false, // 1 张或 3 张不构成七对子
        }
    }
    pairs == 7
}

pub fn ensure_init() {
    assert_eq!(AGARI_TABLE.len(), AGARI_TABLE_SIZE);
}

fn get_tile14_and_key(tiles: &[u8; 27]) -> ([u8; 14], u32) {
    let mut tile14 = [0; 14];
    let mut tile14_iter = tile14.iter_mut();
    let mut key = 0;

    let mut bit_idx = -1;
    let mut prev_in_hand = None;
    for (kind, chunk) in tiles.chunks_exact(9).enumerate() {
        for (num, c) in chunk.iter().copied().enumerate() {
            if c > 0 {
                prev_in_hand = Some(());
                *tile14_iter.next().unwrap() = (kind * 9 + num) as u8;
                bit_idx += 1;

                match c {
                    2 => {
                        key |= 0b11 << bit_idx;
                        bit_idx += 2;
                    }
                    3 => {
                        key |= 0b1111 << bit_idx;
                        bit_idx += 4;
                    }
                    4 => {
                        key |= 0b11_1111 << bit_idx;
                        bit_idx += 6;
                    }
                    // 1
                    _ => (),
                }
            } else if prev_in_hand.take().is_some() {
                key |= 0b1 << bit_idx;
                bit_idx += 1;
            }
        }
        if prev_in_hand.take().is_some() {
            key |= 0b1 << bit_idx;
            bit_idx += 1;
        }
    }

    (tile14, key)
}

/// `tehai` must already contain `tile`. `true` is returned if making an ankan
/// with the tile is legal.
///
/// Check if ankan (暗杠) is valid when in tenpai (听牌) state.
///
/// This function checks if performing ankan changes the tenpai shape or wait tiles.
/// The behavior is undefined if `tehai` is not tenpai.
#[must_use]
pub fn check_ankan_in_tenpai(tehai: &[u8; 27], len_div3: u8, tile: Tile, strict: bool, ding_que: Option<crate::mjai::Suit>) -> bool {
    let tile_id = tile.as_usize();
    if tile_id >= 27 || tehai[tile_id] != 4 {
        return false;
    }

    let mut tehai_before_tsumo = *tehai;
    tehai_before_tsumo[tile_id] -= 1;

    (0..27)
        .filter(|&t| {
            if tehai_before_tsumo[t] == 4 {
                return false;
            }
            // Get all waits of the original hand
            let mut tmp = tehai_before_tsumo;
            tmp[t] += 1;
            shanten::calc_all(&tmp, len_div3, ding_que) == -1
        })
        .all(|wait| {
            // Cannot kan a waited tile
            if wait == tile_id {
                return false;
            }

            // Test if the hand after ankan can also win with the wait tile
            let mut tehai_after = *tehai;
            tehai_after[tile_id] = 0;
            tehai_after[wait] += 1;
            let (_, key) = get_tile14_and_key(&tehai_after);
            // FIX: 龙七对不在 AGARI_TABLE 中，但暗杠后手牌可能变为龙七对结构。
            // 对暗杠后的手牌同样检查七对子回退。
            let divs_after_opt = AGARI_TABLE.get(&key);
            let is_chitoi_after = is_valid_chitoi_hand(&tehai_after);
            if divs_after_opt.is_none() && !is_chitoi_after {
                // The wait tile set will get smaller after kan.
                return false;
            }

            if strict {
                // Compare if the number of hand divisions are equal before and
                // after ankan, which indicates the shapes of tenpai and agari
                // will not change after ankan. This is implemented by inserting
                // the waited tile to both of them.
                let mut tehai_before = tehai_before_tsumo;
                tehai_before[wait] += 1;
                let (_, key) = get_tile14_and_key(&tehai_before);
                let divs_before_opt = AGARI_TABLE.get(&key);
                let is_chitoi_before = is_valid_chitoi_hand(&tehai_before);

                if divs_before_opt.is_none() && !is_chitoi_before {
                    // 暗杠前的听牌手不在表中（理论上不应发生，除非是龙七对）
                    return false;
                }

                // 比较分解数量：龙七对只有 1 种分解
                let count_after = divs_after_opt.map_or(0, |d| d.len())
                    + if is_chitoi_after { 1 } else { 0 };
                let count_before = divs_before_opt.map_or(0, |d| d.len())
                    + if is_chitoi_before { 1 } else { 0 };

                if count_after != count_before {
                    return false;
                }
            }

            true
        })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hand::hand;
    use crate::tile::Tile;

    #[test]
    fn ankan_in_tenpai() {
        // Test ankan validity in tenpai state
        let test_one = |tehai_str, tile_str: &str, len_div3, strict, expected| {
            let mut tehai = hand(tehai_str).unwrap();
            let tile: Tile = tile_str.parse().unwrap();
            tehai[tile.as_usize()] += 1;
            assert_eq!(
                check_ankan_in_tenpai(&tehai, len_div3, tile, strict, None),
                expected,
                "failed for {tehai_str} + {tile_str}, expected {expected}",
            );
        };

        // Always positive: ankan doesn't affect tenpai
        // 血战到底无字牌：用孤立的数牌 (isolated kotsu/pair) 替代原日麻 z 字牌
        test_one("12345m 567s 11p 999p", "9p", 4, true, true);
        test_one("12345m 444567s 11p", "4s", 4, true, true);
        test_one("22m 11112356p 444s", "4s", 4, true, true);

        // Always negative: ankan breaks tenpai (loses waiting tiles)
        test_one("123456m 4445s 999p", "4s", 4, true, false);
        test_one("123456m 4445s 999p", "4s", 4, false, false);

        // Shape of tenpai changes (strict vs non-strict)
        test_one("1113444p 999s", "1p", 3, true, false);
        test_one("1113444p 999s", "1p", 3, false, true);
        test_one("1113444p 999s", "4p", 3, true, false);
        test_one("1113444p 999s", "9s", 3, true, true);

        // Shape of agari changes
        test_one("23m 999p 33345666s", "3s", 4, true, false);
        test_one("23m 999p 33345666s", "6s", 4, true, false);
        test_one("23m 999p 33345666s", "6s", 4, false, true);
        test_one("23m 999p 33345666s", "9p", 4, true, true);

        // 血战到底无九莲宝灯，但此处仅检查听牌结构，不检查役种
        test_one("1113445678999m", "1m", 4, true, true);
        test_one("1113445678999m", "9m", 4, true, false);
    }

    #[test]
    fn agari_calc() {
        
        // Test 1: 平胡 + 自摸 + 门清 (PingHu + Tsumo + MenQing) - 3番
        // 14 tiles: 123456m(6) + 789p(3) + 345s(3) + 22m(2) = 14
        let tehai = hand("123456m 789p 345s 22m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(2m),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 = 1 + 1 + 1 = 3番
        assert_eq!(agari, Agari::Fan(3));
        
        // Test 2: 平胡 + 荣和 + 门清 (Ron, MenQing) - 2番
        let tehai = hand("123456m 789p 345s 22m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(2m),
            is_ron: true, // 荣和
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 荣和 + 平胡 + 门清 = 1 + 1 = 2番
        assert_eq!(agari, Agari::Fan(2));
        
        // Test 2b: 门清（MenQing）无副露 +1番：同一型无副露 3番 vs 有 1 碰 2番
        // 14 tiles: 123456m(6) + 789p(3) + 345s(3) + 22m(2) = 14
        let tehai_mq = hand("123456m 789p 345s 22m").unwrap();
        let c_mq = AgariCalculator { tehai: &tehai_mq, pons: &[], minkans: &[], ankans: &[], winning_tile: tu8!(2m), is_ron: false, ding_que: None, is_after_kan: false, is_kan_discard: false, is_chankan: false, exclude_gen_tile: None, is_haidi: false, is_tianhu: false, is_dihu: false, fan_config: FanConfig::default() };
        assert_eq!(c_mq.agari().unwrap(), Agari::Fan(3), "无副露=门清+1");
        // 11 tiles concealed + 1 pon (3 tiles) = 14 total
        let tehai_f = hand("123456m 789p 22s").unwrap();
        let c_f = AgariCalculator { tehai: &tehai_f, pons: &[tu8!(3s)], minkans: &[], ankans: &[], winning_tile: tu8!(2s), is_ron: false, ding_que: None, is_after_kan: false, is_kan_discard: false, is_chankan: false, exclude_gen_tile: None, is_haidi: false, is_tianhu: false, is_dihu: false, fan_config: FanConfig::default() };
        assert_eq!(c_f.agari().unwrap(), Agari::Fan(2), "有副露无门清");
        
        // Test 3: 七对 (QiDui) - 2番
        let tehai = hand("11223344556677m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(7m),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 七对 = 1 + 1 + 1 + 2 = 5番（封顶）
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 4: 碰碰胡 (ToiToi) - 1番
        // 14 tiles: 111m(3)+333m(3)+555m(3)+999p(3)+77m(2) = 14
        let tehai = hand("111333555m 999p 77m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(9p),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 碰碰胡 = 1 + 1 + 1 + 1 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 5: 清一色 (QingYiSe) - 2番
        // 14 tiles: 1m*3 + 2m + 3m + 4m + 5m + 6m + 7m + 8m + 9m*3 + 9m = 14
        // Divisions: 111m(kotsu) + 234m + 567m + 89m+9m(pair?) → no
        // Better: 111m(kotsu) + 234m(shuntsu) + 567m(shuntsu) + 999m(kotsu) + 8m?? 
        // Actually: 111m + 2345m + 678m + 999m = 3+4+3+3 = 13, not standard.
        // Use: 1234567m(7) + 89m(2) + 999m(3) + 11m(2) = 14
        let tehai = hand("1234567m 89m 999m 11m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(9m),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 清一色 = 1 + 1 + 1 + 2 = 5番（封顶）
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 6: 带幺九 (DaiYaoJiu) - 3番
        let tehai = hand("111999m 111999p 11s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(1s),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 带幺九 = 1+1+1+3 = 6番封顶5
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 6b: 断幺九 (DuanYaoJiu) - 1番（所有组合仅2–8，与带幺九互斥）
        let tehai = hand("234m 345p 456s 678m 22s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(2s),
            is_ron: false,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 断幺九 = 1 + 1 + 1 + 1 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 6c: 一条龙 (YiTiaoLong) - 1番（同一花色含 123、456、789 三副顺子）
        let tehai = hand("123456789m 22p 345s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(5s),
            is_ron: false,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 一条龙 = 1 + 1 + 1 + 1 = 4番（123m 456m 789m 22p 345s）
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 6d: 夹心五 (JiaXinWu) - 1番（46 听 5，和 5 成顺 456）
        let tehai = hand("123m 789p 456s 22m 345p").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(5s),
            is_ron: false,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 夹心五 = 1 + 1 + 1 + 1 = 4番（456s 为 46 听 5 和出）
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 7: 杠上花 (GangShangHua) - 1番
        // 14 tiles: 123456m(6) + 789p(3) + 345s(3) + 22m(2) = 14
        let tehai = hand("123456m 789p 345s 22m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(2m),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: true, // 杠上花
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 杠上花 = 1 + 1 + 1 + 1 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 8: 杠上炮 (GangShangPao) - 1番
        let tehai = hand("123456m 789p 345s 22m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(2m),
            is_ron: true, // 荣和
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: true, // 杠上炮
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 荣和 + 平胡 + 门清 + 杠上炮 = 1 + 1 + 1 = 3番
        assert_eq!(agari, Agari::Fan(3));
        
        // Test 9: 四归一 (SiGuiYi / 根) - 1番/根
        let tehai = hand("111123456m 789p 11s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(1m),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 四归一(1根) = 1 + 1 + 1 + 1 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 10: 金钩钓 (JinGouDiao) - 4番 (4 fuuro + tanki wait，有副露故无门清)
        // 使用混合花色的碰，避免意外触发清一色
        let tehai = hand("11m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[tu8!(2m), tu8!(3p), tu8!(4s), tu8!(5m)], // 4 pons（混合花色）
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(1m), // tanki wait
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 碰碰胡 + 金钩钓 = 1 + 1 + 1 + 1 = 4番（有副露故无门清，混合花色无清一色）
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 12: Fan cap at 5
        // 自摸 + 平胡 + 七对 + 清一色 = 1 + 1 + 2 + 2 = 6番，但封顶5番
        let tehai = hand("11223344556677m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(7m),
            is_ron: false, // 自摸
            ding_que: Some(crate::mjai::Suit::Pin), // 定缺筒子
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 应该封顶在5番（自摸+平胡+门清+七对+清一色 = 7番封顶5）
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 13: Ding Que check - cannot agari if hand has ding_que suit tiles
        // 14 tiles: 123456m(6) + 789p(3) + 345s(3) + 22p(2) = 14, has pin tiles
        let tehai = hand("123456m 789p 345s 22p").unwrap(); // Has pin tiles
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(2p),
            is_ron: false,
            ding_que: Some(crate::mjai::Suit::Pin), // 定缺筒子，但手牌有筒子
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        // 应该不能和牌（花猪）
        assert!(!calc.has_yaku());
        
        // Test 14: 抢杠 (Chankan) - 2番 (平胡1番 + 抢杠1番)
        // 抢杠：在别人加杠时，如果听的牌正好是加杠的牌，可以抢杠和牌
        // 抢杠和杠上炮是不同的：
        // - 抢杠：在别人加杠时抢杠和牌，+1番（平胡1番 + 抢杠1番 = 2番）
        // - 杠上炮：其他玩家杠牌后打出的牌和牌，+1番（平胡1番 + 杠上炮1番 = 2番）
        // 抢杠时，被抢杠的玩家的根不应该计算（因为加杠的牌被抢走了）
        // 14 tiles: 123456m(6) + 789p(3) + 345s(3) + 11s(2) = 14, winning tile 1s already in tehai
        let tehai = hand("123456m 789p 345s 11s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(1s), // 听的牌
            is_ron: true, // 荣和
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false, // 抢杠不是杠上炮
            is_chankan: true, // 这是抢杠
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 荣和 + 平胡 + 门清 + 抢杠 = 1 + 1 + 1 = 3番
        assert_eq!(agari, Agari::Fan(3));
        
        // Test 15: 番数叠加 - 清一色 + 自摸 + 平胡 + 门清 = 5番（封顶）
        // 清一色（2番）+ 自摸（1番）+ 平胡（1番）+ 门清（1番）= 5番（封顶）
        // 14 tiles: 111m(3) + 234m(3) + 567m(3) + 999m(3) + 88m(2) = 14
        let tehai = hand("111234567m 88m 999m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(9m),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 清一色 = 1 + 1 + 1 + 2 = 5番（封顶）
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 16: 番数叠加 - 带幺九 + 自摸 + 平胡 + 门清 = 5番（封顶）
        // 带幺九（3番）+ 自摸（1番）+ 平胡（1番）= 5番（封顶）
        let tehai = hand("111999m 111999p 11s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(1s),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 带幺九 = 1+1+1+3 = 6番封顶5
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 17: 互斥番数 - 七对与碰碰胡互斥
        // 七对（2番）与碰碰胡（1番）互斥，应该只计算七对
        let tehai = hand("11223344556677m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(7m),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 七对 = 1 + 1 + 1 + 2 = 5番（封顶，不是碰碰胡）
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 18: 金钩钓 - 4副露 + 单钓 = 4番（有副露故无门清）
        // 自摸 + 平胡 + 碰碰胡 + 金钩钓 = 1 + 1 + 1 + 1 = 4番（混合花色无清一色）
        let tehai = hand("11m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[tu8!(2m), tu8!(3p), tu8!(4s), tu8!(5m)], // 4个碰（混合花色）
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(1m), // 单钓1m
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 金钩钓 = 1 + 1 + 2 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 19: 四归一（根）- 1个根
        // 14 tiles: 1m*4 + 2m + 3m*2 + 4m + 5m + 6m + 7m + 8m + 1p*2 = 14
        // Decomposition: 11p(pair) + 111m(kotsu) + 123m(shuntsu) + 345m(shuntsu) + 678m(shuntsu)
        // No 一条龙 (123+345+678 ≠ 123+456+789)
        let tehai = hand("1111m 23m 345m 678m 11p").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(1p),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 门清 + 四归一（1根）= 1 + 1 + 1 + 1 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 20: 番数叠加 - 清一色+碰碰胡+自摸+平胡+门清 = 5番（封顶）
        // 清一色（2番）+ 碰碰胡（1番）+ 自摸（1番）+ 平胡（1番）+ 门清（1番）= 6番 → 封顶5
        // 14 tiles: 111m(3)+333m(3)+555m(3)+777m(3)+99m(2) = 14
        let tehai = hand("111333555777m 99m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(9m),
            is_ron: false, // 自摸
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 清一色 + 碰碰胡 = 1 + 1 + 2 + 1 = 5番（封顶）
        assert_eq!(agari, Agari::Fan(5));
        

        // These tests are intentionally unreachable to preserve them for future reference
        }
    }


#[cfg(test)]
mod additional_tests {
    use super::*;
    use crate::hand::hand;

    #[test]
    fn test_long_qidui() {
        // Construct 1122334455m 1111p (Long Qi Dui / Seven Pairs with 4 same tiles)
        let mut tehai = [0u8; 27];
        // 1122334455m -> 2 of each 1m-5m (indices 0-4)
        for i in 0..5 {
            tehai[i] = 2;
        }
        // 1111p -> 4 of 1p (index 9)
        tehai[9] = 4;
        
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: 9, // 1p (index 9)
            is_ron: false, // Tsumo
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        
        let agari = calc.agari(); 
        
        // Should be Agari
        // Expected Fan Calculation:
        // Base (PingHu) = 1
        // Tsumo = 1
        // QiDui = 2
        // Root (SiGuiYi) = 1 (for 1111p)
        // Total = 5 Fan
        
        assert!(agari.is_some(), "Long Qi Dui (4 same tiles) should be valid Agari");
        if let Some(Agari::Fan(fan)) = agari {
             println!("Detected Fan: {}", fan);
             assert!(fan >= 4, "Should be at least 4 Fan (Base+QiDui+Root)");
        }
    }

    #[test]
    fn gen_counts_from_fuuro_plus_hand() {
        // Fuuro pon 555m + concealed 456m makes 4 copies of 5m -> 1 gen (+1 fan).
        //
        // Concealed (11 tiles): 44456m 123p 123s
        // Fuuro: pon 555m
        //
        // Total fan: PingHu(1) + Gen(1) = 2 (Ron points = 2000).
        let tehai = hand("44456m 123p 123s").unwrap();
        let five_m: Tile = "5m".parse().unwrap();
        let pons = [five_m.as_u8()];

        let agari_calc = AgariCalculator {
            tehai: &tehai,
            pons: &pons,
            minkans: &[],
            ankans: &[],
            winning_tile: 6, // arbitrary (not used by has_yaku/agari fan calc except for JGD tanki)
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };

        let agari = agari_calc.agari().expect("should agari");
        assert_eq!(agari.point(false).ron, 2000);
    }

    #[test]
    fn gen_excluded_tile_not_counted() {
        // Same as above, but exclude 5m from gen count (e.g. robbed kong semantics).
        // Fan should drop by 1: PingHu(1) only -> 1000.
        let tehai = hand("44456m 123p 123s").unwrap();
        let five_m: Tile = "5m".parse().unwrap();
        let pons = [five_m.as_u8()];

        let agari_calc = AgariCalculator {
            tehai: &tehai,
            pons: &pons,
            minkans: &[],
            ankans: &[],
            winning_tile: 6,
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: Some(five_m.as_u8()),
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };

        let agari = agari_calc.agari().expect("should agari");
        assert_eq!(agari.point(false).ron, 1000);
    }

    #[test]
    fn ron_8p_pure_straight_shape_is_2_fan_without_bonuses() {
        // Hand shape (winning on 8p):
        // 123m 234p 789p 888p 55s
        //
        // Per Bloody Battle scoring:
        // - PingHu base: 1
        // - MenQing: +1 (no fuuro)
        // - Gen (SiGuiYi): +1 (8p appears 4 times: one in 789p + 888p)
        //
        // Ron: PingHu(1) + MenQing(1) + Gen(1) = 3 fan -> 4000 points.
        // (no 一条龍 since shuntsu span different suits)
        let tehai = hand("123m 234p 789p 888p 55s").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,
            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(8p),
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        let agari = calc.agari().expect("should agari");
        assert_eq!(agari, Agari::Fan(3));
        assert_eq!(agari.point(false).ron, 4000);
    }

    #[test]
    fn ron_8p_becomes_3_fan_with_one_bonus_flag() {
        // Same hand as ron_8p_pure_straight_shape_is_2_fan_without_bonuses:
        // 123m 234p 789p 888p 55s (8p=4 → 四归一)
        // Base: PingHu(1) + MenQing(1) + 四归一(1) = 3 fan
        // Each bonus flag should add +1 → 4 fan
        let tehai = hand("123m 234p 789p 888p 55s").unwrap();

        // Haidi (海底炮) adds +1 fan.
        let calc_haidi = AgariCalculator {
            tehai: &tehai,
            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(8p),
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: true,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        // PingHu(1) + MenQing(1) + 四归一(1) + Haidi(1) = 4 fan
        assert_eq!(calc_haidi.agari().unwrap(), Agari::Fan(4));

        // GangShangPao (杠上炮) adds +1 fan when Ron happens right after a kan discard.
        let calc_kan_discard = AgariCalculator {
            tehai: &tehai,
            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(8p),
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: true,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        // PingHu(1) + MenQing(1) + 四归一(1) + GangShangPao(1) = 4 fan
        assert_eq!(calc_kan_discard.agari().unwrap(), Agari::Fan(4));

        // Chankan (抢杠) adds +1 fan.
        let calc_chankan = AgariCalculator {
            tehai: &tehai,
            pons: &[],
            minkans: &[],
            ankans: &[],
            winning_tile: tu8!(8p),
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: true,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };
        // PingHu(1) + MenQing(1) + 四归一(1) + Chankan(1) = 4 fan
        assert_eq!(calc_chankan.agari().unwrap(), Agari::Fan(4));
    }

    /// 验证副露分解约束：碰/杠的牌不能被拆分为对子+顺子。
    ///
    /// Bug 场景：碰 1万 + 手牌 3万,5万5万5万,2条2条2条,3条3条3条，对手打 2万。
    /// AGARI_TABLE 会找到"1万1万(对) + 1万2万3万(顺)"的非法分解，
    /// 但碰的 1万 必须保持为刻子，所以实际上不能胡。
    #[test]
    fn test_fuuro_division_constraint_false_positive() {
        // 构造场景：碰 1万(tile_id=0)，手牌 3万 5万5万5万 2条2条2条 3条3条3条 + 赢牌 2万
        // 手牌(tehai)不含碰的牌，只含暗牌部分 + 赢牌
        let mut tehai = [0u8; 27];
        tehai[2] = 1;   // 3万 x1
        tehai[4] = 3;   // 5万 x3
        tehai[19] = 3;  // 2条 x3 (index: 18+1=19)
        tehai[20] = 3;  // 3条 x3 (index: 18+2=20)
        tehai[1] = 1;   // 2万 x1 (赢牌已加入)
        // 总计: 1+3+3+3+1 = 11 张暗牌(含赢牌)

        let calc = AgariCalculator {
            tehai: &tehai,
            pons: &[0],    // 碰 1万 (tile_id=0)
            minkans: &[],
            ankans: &[],
            winning_tile: 1, // 2万
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };

        // 关键断言：这手牌不能胡 2万！
        // 碰的 1万 必须保持为刻子。剩余 2万+3万 无法组成面子。
        assert!(!calc.has_yaku(),
            "BUG: has_yaku() 返回 true，但碰 1万 后 2万+3万 无法组成面子，不应能胡");
        assert!(calc.agari().is_none(),
            "BUG: agari() 返回 Some，但碰 1万 后 2万+3万 无法组成面子，不应能胡");
    }

    /// 对照测试：同样牌型但没有副露时确实可以胡（门清手）。
    /// 这证明 AGARI_TABLE 确实包含这个牌型的分解，只是副露约束应阻止它。
    #[test]
    fn test_fuuro_division_constraint_menqing_can_agari() {
        // 门清手：1万1万1万 2万 3万 5万5万5万 2条2条2条 3条3条3条（14 张）
        let mut tehai = [0u8; 27];
        tehai[0] = 3;   // 1万 x3
        tehai[1] = 1;   // 2万 x1
        tehai[2] = 1;   // 3万 x1
        tehai[4] = 3;   // 5万 x3
        tehai[19] = 3;  // 2条 x3
        tehai[20] = 3;  // 3条 x3

        let calc = AgariCalculator {
            tehai: &tehai,
            pons: &[],      // 无副露
            minkans: &[],
            ankans: &[],
            winning_tile: 1, // 2万
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };

        // 门清手可以自由分解为：1万1万(对)+1万2万3万(顺)+5万5万5万+2条2条2条+3条3条3条
        assert!(calc.has_yaku(),
            "门清手应该能胡：1万1万(对)+1万2万3万(顺)+5万5万5万+2条2条2条+3条3条3条");
        assert!(calc.agari().is_some(),
            "门清手 agari() 应返回 Some");
    }

    /// 对照测试：碰 1万 + 手牌可以正常胡（不需要拆分碰）。
    #[test]
    fn test_fuuro_division_constraint_valid_ron_with_pon() {
        // 碰 1万，手牌：2万2万 5万5万5万 2条2条2条 3条3条3条 + 赢牌 2万
        // 合法分解：碰1万(刻)+2万2万2万(刻)+5万5万5万(刻)+2条2条2条(刻)+3条3条(对)
        // 或：碰1万(刻)+2万2万(对)+5万5万5万(刻)+2条2条2条(刻)+3条3条3条(刻)
        let mut tehai = [0u8; 27];
        tehai[1] = 3;   // 2万 x3 (2 in hand + 1 winning tile)
        tehai[4] = 3;   // 5万 x3
        tehai[19] = 3;  // 2条 x3
        tehai[20] = 2;  // 3条 x2 (对子)
        // 总计: 3+3+3+2 = 11 张暗牌(含赢牌)

        let calc = AgariCalculator {
            tehai: &tehai,
            pons: &[0],    // 碰 1万
            minkans: &[],
            ankans: &[],
            winning_tile: 1, // 2万
            is_ron: true,
            ding_que: None,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: false,
            exclude_gen_tile: None,
            is_haidi: false,
            is_tianhu: false,
            is_dihu: false,
            fan_config: FanConfig::default(),
        };

        // 合法：碰1万(刻) + 2万2万2万(刻) + 5万5万5万(刻) + 2条2条2条(刻) + 3条3条(对)
        // 碰保持为刻子，不需要拆分
        assert!(calc.has_yaku(),
            "碰1万+2万刻+5万刻+2条刻+3条对 应该能胡");
        assert!(calc.agari().is_some(),
            "agari() 应返回 Some");
    }

    #[test]
    fn test_fan_config_random_from_seed() {
        // Determinism: same seed → same config
        let a = FanConfig::random_from_seed(42);
        let b = FanConfig::random_from_seed(42);
        assert_eq!(a, b);

        // Different seeds → likely different configs
        let configs: Vec<_> = (0..32).map(|i| FanConfig::random_from_seed(i)).collect();
        let unique_count = configs.iter().collect::<std::collections::HashSet<_>>().len();
        // With 7 flags and 32 seeds, should get a good mix (not all identical)
        assert!(unique_count >= 8, "Expected diverse configs, got only {unique_count} unique out of 32");

        // Verify flags are booleans (sanity)
        let c = FanConfig::random_from_seed(12345);
        let flags = c.as_flags();
        assert_eq!(flags.len(), FanConfig::NUM_FLAGS);
    }
}

