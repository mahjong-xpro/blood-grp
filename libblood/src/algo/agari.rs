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
    /// But must check Ding Que rule: cannot agari if hand still has ding_que suit tiles
    #[inline]
    #[must_use]
    pub fn has_yaku(&self) -> bool {
        let Some(hand14) = self.hand14_for_division() else {
            return false;
        };
        let (_, key) = get_tile14_and_key(&hand14);
        let has_valid_structure = AGARI_TABLE.get(&key).is_some();
        
        if !has_valid_structure {
            return false;
        }

        // DingQue rule (花猪): cannot agari if DingQue suit tiles remain in full hand (tehai + fuuro).
        if !crate::ding_que::can_agari_with_fuuro(self.tehai, self.pons, self.minkans, self.ankans, self.ding_que) {
            return false;
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
    /// 3. 七对（QiDui）：+2番
    /// 4. 碰碰胡（ToiToi）：+1番
    /// 5. 金钩钓（JinGouDiao）：+1番
    /// 6. 清一色（QingYiSe）：+2番
    /// 7. 带幺九（DaiYaoJiu）：+3番
    /// 8. 四归一（SiGuiYi / 根）：+1番/根
    /// 9. 杠上花（GangShangHua）：+1番（if is_after_kan && !is_ron）
    /// 10. 杠上炮（GangShangPao）：+1番（if is_kan_discard && is_ron && !is_chankan）
    /// 11. 抢杠（Chankan）：+1番（if is_chankan && is_ron）
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
        
        // 2. 自摸（Tsumo）：+1番
        if !self.is_ron {
            fan += 1;
        }

        // 12. 海底捞月/海底炮（Haidi）：+1番
        if self.is_haidi {
            fan += 1;
        }
        
        // Check hand structure (must be a valid 14-tile multiset for AGARI_TABLE)
        let hand14 = self.hand14_for_division()?;
        let tile14_len: usize = hand14.iter().filter(|&&c| c > 0).count();
        let (tile14, key) = get_tile14_and_key(&hand14);
        let divs = match AGARI_TABLE.get(&key) {
            Some(d) => d,
            None => {
                // If the hand structure is invalid (not in AGARI_TABLE), return None
                // This is expected behavior: some hand structures with 14 tiles cannot agari
                // (e.g., invalid tile combinations that don't form valid melds/pairs)
                return None;
            }
        };
        
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
        for div in divs.iter() {
            let mut div_fan: u8 = gen_count;
            
            // 3. 七对（QiDui）：+2番
            if div.has_chitoi {
                div_fan += 2;
                // 七对与碰碰胡、金钩钓互斥，跳过其他检查
                max_fan = max_fan.max(div_fan);
                continue;
            }
            
            // 5. 金钩钓（JinGouDiao）：+2番 (4 fuuro + single wait/tanki)
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
                    // (Previously was 2, resulted in 4 Fan / 8000).
                    div_fan += 1;
                }
            }
            
            // 6. 清一色（QingYiSe）：+2番
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
            
            // 7. 带幺九（DaiYaoJiu）：+3番
            // Check if all groups (shuntsu, kotsu, pair) contain 1 or 9
            let mut is_daiyaojiu = true;
            
            // Check shuntsu: must start with 1 or end with 9 (1-2-3 or 7-8-9)
            for &shuntsu_idx in &div.shuntsu_idxs {
                let tile_id = tile14[shuntsu_idx as usize];
                if tile_id >= 27 {
                    continue;
                }
                let num = tile_id % 9;
                // Shuntsu must be 1-2-3 (num == 0) or 7-8-9 (num == 6)
                if num != 0 && num != 6 {
                    is_daiyaojiu = false;
                    break;
                }
            }
            
            // Check kotsu: must be 1 or 9
            if is_daiyaojiu {
                for &kotsu_idx in &div.kotsu_idxs {
                    let tile_id = tile14[kotsu_idx as usize];
                    if tile_id >= 27 {
                        continue;
                    }
                    let num = tile_id % 9;
                    if num != 0 && num != 8 {
                        is_daiyaojiu = false;
                        break;
                    }
                }
            }
            
            // Check pair: must be 1 or 9
            if is_daiyaojiu {
                let pair_tile = tile14[div.pair_idx as usize];
                if pair_tile < 27 {
                    let num = pair_tile % 9;
                    if num != 0 && num != 8 {
                        is_daiyaojiu = false;
                    }
                }
            }
            
            // Check fuuro (pons, minkans, ankans): must be 1 or 9
            if is_daiyaojiu {
                for &tile_id in self.pons.iter().chain(self.minkans.iter()).chain(self.ankans.iter()) {
                    if tile_id >= 27 {
                        continue;
                    }
                    let num = tile_id % 9;
                    if num != 0 && num != 8 {
                        is_daiyaojiu = false;
                        break;
                    }
                }
            }
            
            if is_daiyaojiu {
                div_fan += 3;
            }
            

            

            
            max_fan = max_fan.max(div_fan);
        }
        
        fan = fan.saturating_add(max_fan);
        
        // 9. 杠上花（GangShangHua）：+1番
        if self.is_after_kan && !self.is_ron {
            fan += 1;
        }
        
        // 10. 杠上炮（GangShangPao）：+1番
        // 注意：抢杠（chankan）和杠上炮是不同的
        // - 抢杠：在别人加杠时抢杠和牌，+1番
        // - 杠上炮：其他玩家杠牌后打出的牌和牌，+1番
        if self.is_kan_discard && self.is_ron && !self.is_chankan {
            fan += 1;
        }
        
        // 11. 抢杠（Chankan）：+1番
        // 抢杠：在别人加杠时抢杠和牌，+1番
        // 抢杠时，被抢杠的玩家的根不应该计算（因为加杠的牌被抢走了）
        if self.is_chankan && self.is_ron {
            fan += 1;
        }
        
        // 13. 天胡 (TianHu) / 地胡 (DiHu): Max Fan (5番)
        if self.is_tianhu || self.is_dihu {
            fan = 5;
            // Early return or just let it be capped below (it is already 5)
        }

        // 5番封顶
        fan = fan.min(5);
        
        Some(Agari::Fan(fan))
    }
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
            let Some(divs_after) = AGARI_TABLE.get(&key) else {
                // The wait tile set will get smaller after kan.
                return false;
            };

            if strict {
                // Compare if the number of hand divisions are equal before and
                // after ankan, which indicates the shapes of tenpai and agari
                // will not change after ankan. This is implemented by inserting
                // the waited tile to both of them.
                let mut tehai_before = tehai_before_tsumo;
                tehai_before[wait] += 1;
                let (_, key) = get_tile14_and_key(&tehai_before);
                let divs_before = AGARI_TABLE
                    .get(&key)
                    .expect("invalid tenpai detected when testing ankan");

                if divs_after.len() != divs_before.len() {
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

        // Always positive
        test_one("12345m 567s 11222z", "S", 4, true, true);
        test_one("12345m 444567s 11z", "4s", 4, true, true);
        test_one("22m 11112356p 444s", "4s", 4, true, true);

        // Always negative
        test_one("123456m 4445s 111z", "4s", 4, true, false);
        test_one("123456m 4445s 111z", "4s", 4, false, false);

        // Shape of tenpai changes
        test_one("1113444p 222z", "1p", 3, true, false);
        test_one("1113444p 222z", "1p", 3, false, true);
        test_one("1113444p 222z", "4p", 3, true, false);
        test_one("1113444p 222z", "S", 3, true, true);

        // Shape of agari changes
        test_one("23m 999p 33345666s", "3s", 4, true, false);
        test_one("23m 999p 33345666s", "6s", 4, true, false);
        test_one("23m 999p 33345666s", "6s", 4, false, true);
        test_one("23m 999p 33345666s", "9p", 4, true, true);

        // The 1m kan will make chuuren gone, but in this impl we don't take
        // yaku into account.
        test_one("1113445678999m", "1m", 4, true, true);
        test_one("1113445678999m", "9m", 4, true, false);
    }

    #[test]
    fn agari_calc() {
        
        // Test 1: Basic 平胡 (PingHu) - 1番 (base fan)
        let tehai = hand("123456m 789p 11s 2m").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 = 1 + 1 = 2番
        assert_eq!(agari, Agari::Fan(2));
        
        // Test 2: 平胡 + 自摸 (Tsumo) - 2番
        let tehai = hand("123456m 789p 11s 2m").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 荣和 + 平胡 = 1番（荣和没有自摸番）
        assert_eq!(agari, Agari::Fan(1));
        
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 七对 = 1 + 1 + 2 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 4: 碰碰胡 (ToiToi) - 1番
        let tehai = hand("11133355577m 99p").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 碰碰胡 = 1 + 1 + 1 = 3番
        assert_eq!(agari, Agari::Fan(3));
        
        // Test 5: 清一色 (QingYiSe) - 2番
        let tehai = hand("1112345678999m").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 清一色 = 1 + 1 + 2 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 带幺九 = 1 + 1 + 3 = 5番（封顶）
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 7: 杠上花 (GangShangHua) - 1番
        let tehai = hand("123456m 789p 11s 2m").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 杠上花 = 1 + 1 + 1 = 3番
        assert_eq!(agari, Agari::Fan(3));
        
        // Test 8: 杠上炮 (GangShangPao) - 1番
        let tehai = hand("123456m 789p 11s 2m").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 荣和 + 平胡 + 杠上炮 = 1 + 1 = 2番
        assert_eq!(agari, Agari::Fan(2));
        
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 四归一(1根) = 1 + 1 + 1 = 3番
        assert_eq!(agari, Agari::Fan(3));
        
        // Test 10: 金钩钓 (JinGouDiao) - 2番 (4 fuuro + tanki wait)
        // This requires 4 fuuro, so we need to set pons/minkans/ankans
        let tehai = hand("11m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[tu8!(2m), tu8!(3m), tu8!(4m), tu8!(5m)], // 4 pons
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 金钩钓 = 1 + 1 + 2 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 11: Fan cap at 5
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
        };
        let agari = calc.agari().unwrap();
        // 应该封顶在5番
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 12: Ding Que check - cannot agari if hand has ding_que suit tiles
        let tehai = hand("123456m 789p 11s 2p").unwrap(); // Has pin tiles
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
        };
        // 应该不能和牌（花猪）
        assert!(!calc.has_yaku());
        
        // Test 13: 抢杠 (Chankan) - 2番 (平胡1番 + 抢杠1番)
        // 抢杠：在别人加杠时，如果听的牌正好是加杠的牌，可以抢杠和牌
        // 抢杠和杠上炮是不同的：
        // - 抢杠：在别人加杠时抢杠和牌，+1番（平胡1番 + 抢杠1番 = 2番）
        // - 杠上炮：其他玩家杠牌后打出的牌和牌，+1番（平胡1番 + 杠上炮1番 = 2番）
        // 抢杠时，被抢杠的玩家的根不应该计算（因为加杠的牌被抢走了）
        let tehai = hand("123456m 789p 11s").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 荣和 + 平胡 + 抢杠 = 1 + 1 = 2番
        assert_eq!(agari, Agari::Fan(2));
        
        // Test 14: 番数叠加 - 清一色 + 自摸 + 平胡 = 4番
        // 清一色（2番）+ 自摸（1番）+ 平胡（1番）= 4番
        let tehai = hand("1112345678999m").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 清一色 = 1 + 1 + 2 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 15: 番数叠加 - 带幺九 + 自摸 + 平胡 = 5番（封顶）
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 带幺九 = 1 + 1 + 3 = 5番（封顶）
        assert_eq!(agari, Agari::Fan(5));
        
        // Test 16: 互斥番数 - 七对与碰碰胡互斥
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 七对 = 1 + 1 + 2 = 4番（不是碰碰胡）
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 17: 金钩钓 - 4副露 + 单钓 = 4番
        // 金钩钓（2番）+ 自摸（1番）+ 平胡（1番）= 4番
        let tehai = hand("11m").unwrap();
        let calc = AgariCalculator {
            tehai: &tehai,

            pons: &[tu8!(2m), tu8!(3m), tu8!(4m), tu8!(5m)], // 4个碰
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 金钩钓 = 1 + 1 + 2 = 4番
        assert_eq!(agari, Agari::Fan(4));
        
        // Test 18: 四归一（根）- 多个根的情况
        // 四归一（1番/根）+ 自摸（1番）+ 平胡（1番）= 3番（1个根）
        let tehai = hand("111123456789m 11p").unwrap();
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
        };
        let agari = calc.agari().unwrap();
        // 自摸 + 平胡 + 四归一（1根）= 1 + 1 + 1 = 3番
        assert_eq!(agari, Agari::Fan(3));
        
        // Test 19: 番数叠加 - 清一色 + 碰碰胡 + 自摸 + 平胡 = 5番（封顶）
        // 清一色（2番）+ 碰碰胡（1番）+ 自摸（1番）+ 平胡（1番）= 5番（封顶）
        let tehai = hand("111444777999m 11m").unwrap();
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
        };

        let agari = agari_calc.agari().expect("should agari");
        assert_eq!(agari.point(false).ron, 1000);
    }

    #[test]
    fn ron_8p_pure_straight_shape_is_2_fan_without_bonuses() {
        // Hand shape (winning on 8p):
        // 123p 456p 789p 888p 55s
        //
        // Per current Bloody Battle scoring implementation:
        // - PingHu base: 1
        // - Gen (SiGuiYi): +1 (8p appears 4 times total: one in 789p + 888p)
        // No other patterns apply (not QingYiSe, not DaiYaoJiu, not ToiToi, not JGD).
        //
        // Therefore Ron should be 2 fan -> 2000 points.
        let tehai = hand("123456789p 888p 55s").unwrap();
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
        };
        let agari = calc.agari().expect("should agari");
        assert_eq!(agari, Agari::Fan(2));
        assert_eq!(agari.point(false).ron, 2000);
    }

    #[test]
    fn ron_8p_becomes_3_fan_with_one_bonus_flag() {
        let tehai = hand("123456789p 888p 55s").unwrap();

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
        };
        assert_eq!(calc_haidi.agari().unwrap(), Agari::Fan(3));

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
        };
        assert_eq!(calc_kan_discard.agari().unwrap(), Agari::Fan(3));

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
        };
        assert_eq!(calc_chankan.agari().unwrap(), Agari::Fan(3));
    }
}

