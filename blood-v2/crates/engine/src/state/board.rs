use crate::consts::*;
use crate::tile::*;
use crate::hand::*;
use crate::algo::agari::{WinContext, FanConfig, calc_fan};
use crate::algo::point::calc_score;
use crate::algo::shanten::{calc_shanten, waiting_tiles};
use super::player::PlayerState;
use super::action::{Action, ActionCandidate};
use super::event::Event;
use super::ding_que;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    DingQue,
    SelfCheck,
    KanSelect,
    Discard,
    Reaction,
    Scoring,
    Done,
}

/// Full board state for a single game of Bloody Battle Mahjong
#[derive(Debug, Clone)]
pub struct BoardState {
    pub phase: Phase,
    pub players: [PlayerState; NUM_PLAYERS],
    pub wall: Vec<Tile>,
    pub wall_idx: usize,
    pub wall_back_idx: usize, // for kan draws from back
    pub dealer: usize,
    pub current_player: usize,
    pub turn_count: u16,
    pub win_count: u8,

    // Last discard tracking
    pub last_discard: Option<(usize, Tile)>,
    pub last_discard_is_kan: bool,

    // Reaction collection
    pub reactions: [Option<Action>; NUM_PLAYERS],
    pub reaction_pending: [bool; NUM_PLAYERS],

    // Event log
    pub events: Vec<Event>,

    // Fan config
    pub fan_config: FanConfig,

    // Turn 0 tracking for tianhu/dihu
    pub dahai_count: u16,

    // Configurable initial score (stored for obs encoding / header generation)
    pub initial_score: i32,
}

impl BoardState {
    /// Create a new board with the default initial score (100,000).
    pub fn new(seed: u64) -> Self {
        Self::with_initial_score(seed, INITIAL_SCORE)
    }

    /// Create a new board with a custom initial score per player.
    pub fn with_initial_score(seed: u64, initial_score: i32) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let wall = generate_deck(&mut rng);
        let dealer = (seed % NUM_PLAYERS as u64) as usize;

        let mut state = Self {
            phase: Phase::DingQue,
            players: std::array::from_fn(|_| PlayerState::with_score(initial_score)),
            wall,
            wall_idx: 0,
            wall_back_idx: 0,
            dealer,
            current_player: dealer,
            turn_count: 0,
            win_count: 0,
            last_discard: None,
            last_discard_is_kan: false,
            reactions: [None; NUM_PLAYERS],
            reaction_pending: [false; NUM_PLAYERS],
            events: Vec::new(),
            fan_config: FanConfig::default(),
            dahai_count: 0,
            initial_score,
        };

        // Deal 13 tiles to each player
        for p in 0..NUM_PLAYERS {
            for _ in 0..HAND_SIZE {
                let tile = state.draw_from_wall();
                add_tile(&mut state.players[p].hand, tile);
                state.players[p].see_tile(tile);
            }
        }
        // Dealer gets 14th tile
        let extra = state.draw_from_wall();
        add_tile(&mut state.players[dealer].hand, extra);
        state.players[dealer].last_drawn_tile = Some(extra);
        state.players[dealer].see_tile(extra);

        // Record initial hands as Deal events (for replay)
        for p in 0..NUM_PLAYERS {
            let tiles: Vec<Tile> = state.players[p].hand.iter().enumerate()
                .flat_map(|(tile, &count)| std::iter::repeat(tile as Tile).take(count as usize))
                .collect();
            state.events.push(Event::Deal { player: p, tiles });
        }

        state.wall_back_idx = state.wall.len();

        state
    }

    pub fn wall_remaining(&self) -> usize {
        if self.wall_back_idx > self.wall_idx {
            self.wall_back_idx - self.wall_idx
        } else {
            0
        }
    }

    fn draw_from_wall(&mut self) -> Tile {
        debug_assert!(self.wall_idx < self.wall.len(), "draw_from_wall: wall exhausted");
        let t = self.wall[self.wall_idx];
        self.wall_idx += 1;
        t
    }

    fn draw_from_back(&mut self) -> Tile {
        debug_assert!(self.wall_back_idx > self.wall_idx, "draw_from_back: wall exhausted");
        self.wall_back_idx -= 1;
        self.wall[self.wall_back_idx]
    }

    pub fn is_done(&self) -> bool {
        self.phase == Phase::Done
    }

    pub fn active_player_count(&self) -> usize {
        self.players.iter().filter(|p| !p.has_won).count()
    }

    /// Whether all players have completed ding_que selection.
    pub fn all_ding_que_done(&self) -> bool {
        self.players.iter().all(|p| p.ding_que.is_some())
    }

    /// Per-player ding_que completion status, derived from `players[i].ding_que`.
    pub fn ding_que_done(&self) -> [bool; NUM_PLAYERS] {
        std::array::from_fn(|i| self.players[i].ding_que.is_some())
    }

    /// Get the decision request for the current state
    pub fn get_decision_request(&self, player_id: usize) -> Option<ActionCandidate> {
        match self.phase {
            Phase::DingQue => {
                if self.players[player_id].ding_que.is_none() {
                    Some(ActionCandidate {
                        can_ding_que: true,
                        ..Default::default()
                    })
                } else {
                    None
                }
            }
            Phase::SelfCheck if self.current_player == player_id => {
                self.get_self_check_actions(player_id)
            }
            Phase::KanSelect if self.current_player == player_id => {
                let p = &self.players[player_id];
                let ankan = p.can_ankan_tiles();
                let kakan = p.can_kakan_tiles();
                let mut kan_tiles = ankan;
                kan_tiles.extend(kakan);
                Some(ActionCandidate {
                    at_kan_select: true,
                    kan_tiles,
                    ..Default::default()
                })
            }
            Phase::Discard if self.current_player == player_id => {
                let p = &self.players[player_id];
                let candidates = p.discard_candidates();
                Some(ActionCandidate {
                    can_discard: true,
                    discard_tiles: candidates,
                    ..Default::default()
                })
            }
            Phase::Reaction if self.reaction_pending[player_id] => {
                self.get_reaction_actions(player_id)
            }
            _ => None,
        }
    }

    fn get_self_check_actions(&self, player_id: usize) -> Option<ActionCandidate> {
        let p = &self.players[player_id];
        if p.has_won { return None; }
        let shanten = calc_shanten(&p.hand, p.melds.len());

        let mut candidate = ActionCandidate::default();

        // Tsumo check
        if shanten == -1 && p.ding_que_completed() {
            candidate.can_agari = true;
        }

        // AnKan check
        let ankan_tiles = p.can_ankan_tiles();
        let kakan_tiles = p.can_kakan_tiles();

        if !ankan_tiles.is_empty() || !kakan_tiles.is_empty() {
            let mut kan_tiles = ankan_tiles;
            kan_tiles.extend(kakan_tiles);
            candidate.can_kan = true;
            candidate.kan_tiles = kan_tiles;
        }

        // If can agari or kan, add pass option too (to decline and just discard)
        if candidate.can_agari || candidate.can_kan {
            candidate.can_pass = true;
            return Some(candidate);
        }

        // Otherwise go straight to discard
        None
    }

    fn get_reaction_actions(&self, player_id: usize) -> Option<ActionCandidate> {
        let (discarder, tile) = self.last_discard?;
        if discarder == player_id { return None; }
        let p = &self.players[player_id];
        if p.has_won { return None; }

        let mut candidate = ActionCandidate {
            can_pass: true,
            ..Default::default()
        };

        // Ron check
        // Blood mahjong has no furiten rule; only 过手加番 applies.
        if p.ding_que_completed() {
            let mut h = p.hand;
            add_tile(&mut h, tile);
            if is_complete(&h, p.melds.len()) {
                if let Some(passed_fan) = p.furiten_passed_ron_fan {
                    // 过手加番: passed on this tile before; allow ron only if fan increased
                    let ctx = self.make_win_context(player_id, tile, true);
                    if let Some(result) = calc_fan(&ctx) {
                        if result.fan > passed_fan {
                            candidate.can_agari = true;
                        }
                    }
                } else {
                    candidate.can_agari = true;
                }
            }
        }

        // Pon check
        if p.hand[tile as usize] >= 2 && !ding_que::is_ding_que_tile(p.ding_que, tile) {
            candidate.can_pon = true;
        }

        // MinKan check
        if p.hand[tile as usize] >= 3 && !ding_que::is_ding_que_tile(p.ding_que, tile) {
            candidate.can_kan = true;
            candidate.kan_tiles = vec![tile];
        }

        if candidate.can_agari || candidate.can_pon || candidate.can_kan {
            Some(candidate)
        } else {
            None
        }
    }

    /// Apply an action from a player
    pub fn apply_action(&mut self, player_id: usize, action: Action) {
        match self.phase {
            Phase::DingQue => self.apply_ding_que(player_id, action),
            Phase::SelfCheck => self.apply_self_check(player_id, action),
            Phase::KanSelect => self.apply_kan_select(player_id, action),
            Phase::Discard => self.apply_discard(player_id, action),
            Phase::Reaction => self.apply_reaction(player_id, action),
            _ => {}
        }
    }

    fn apply_ding_que(&mut self, player_id: usize, action: Action) {
        if let Action::DingQue(suit) = action {
            // Guard: ignore duplicate ding_que for the same player
            if self.players[player_id].ding_que.is_some() { return; }
            self.players[player_id].ding_que = Some(suit);
            self.events.push(Event::DingQue { player: player_id, suit });

            if self.all_ding_que_done() {
                self.current_player = self.dealer;
                self.phase = Phase::SelfCheck;
                if self.get_self_check_actions(self.dealer).is_none() {
                    self.phase = Phase::Discard;
                }
            }
        }
    }

    fn apply_self_check(&mut self, player_id: usize, action: Action) {
        // Guard: only the current player can act in SelfCheck.
        // Applying another player's action here (e.g. after a stall break) would
        // read last_drawn_tile for the wrong player and panic on Agari.
        if player_id != self.current_player {
            return;
        }
        match action {
            Action::Agari => {
                let tile = self.players[player_id].last_drawn_tile
                    .expect("tsumo declared but no drawn tile recorded");
                self.events.push(Event::Tsumo { player: player_id, tile });
                self.process_win(player_id, None, tile);
            }
            Action::Kan => {
                let p = &self.players[player_id];
                let ankan = p.can_ankan_tiles();
                let kakan = p.can_kakan_tiles();
                let total = ankan.len() + kakan.len();

                if total == 1 {
                    let (tile, is_ankan) = if !ankan.is_empty() {
                        (ankan[0], true)
                    } else {
                        (kakan[0], false)
                    };
                    self.execute_kan(player_id, tile, is_ankan);
                } else if total > 1 {
                    self.phase = Phase::KanSelect;
                } else {
                    self.phase = Phase::Discard;
                }
            }
            // Discard action in self_check phase: model chose a tile instead of Pass.
            // Treat as Pass — the actual discard happens in the Discard phase.
            // Any other unrecognised action (e.g. Pon output by a confused model)
            // is also treated as Pass to avoid a no-op stall.
            _ => {
                self.phase = Phase::Discard;
            }
        }
    }

    fn apply_kan_select(&mut self, player_id: usize, action: Action) {
        if player_id != self.current_player {
            return;
        }
        if let Action::Discard(tile) = action {
            let p = &self.players[player_id];
            // Determine kan type directly from hand state — avoids recomputing tile lists.
            // AnKan: 4 copies in hand (ding_que suit excluded by can_ankan_tiles logic).
            // KaKan: has a Pon meld for this tile and ≥1 copy in hand.
            let is_ankan = p.hand[tile as usize] >= 4
                && !ding_que::is_ding_que_tile(p.ding_que, tile);
            let is_kakan = !is_ankan
                && p.hand[tile as usize] >= 1
                && p.melds.iter().any(|m| matches!(m, MeldType::Pon(t) if *t == tile));
            if is_ankan || is_kakan {
                self.execute_kan(player_id, tile, is_ankan);
                return;
            }
        }
        // Invalid action or unrecognized tile — fall back to discard phase
        self.phase = Phase::Discard;
    }

    fn execute_kan(&mut self, player_id: usize, tile: Tile, is_ankan: bool) {
        let p = &mut self.players[player_id];

        if is_ankan {
            // AnKan — guard: need 4 copies in hand
            if p.hand[tile as usize] < 4 {
                // Stale action: hand no longer has 4 copies. Fall back to discard phase.
                self.phase = Phase::Discard;
                return;
            }
            // AnKan
            remove_tile(&mut p.hand, tile);
            remove_tile(&mut p.hand, tile);
            remove_tile(&mut p.hand, tile);
            remove_tile(&mut p.hand, tile);
            p.melds.push(MeldType::AnKan(tile));
            p.meld_from.push(None); // 暗杠无来源玩家

            for i in 0..NUM_PLAYERS {
                if i != player_id && !self.players[i].has_won {
                    self.players[i].see_tile_n(tile, 4);
                }
            }

            self.events.push(Event::AnKan { player: player_id, tile });

            // AnKan payment: each non-winner pays 2000
            for i in 0..NUM_PLAYERS {
                if i != player_id && !self.players[i].has_won {
                    self.players[i].score -= 2000;
                    self.players[player_id].score += 2000;
                    self.events.push(Event::KanPayment {
                        payer: i, receiver: player_id, amount: 2000,
                    });
                }
            }
        } else {
            // KaKan — guard: need at least 1 copy in hand + matching Pon meld
            if p.hand[tile as usize] < 1 || !p.melds.iter().any(|m| matches!(m, MeldType::Pon(t) if *t == tile)) {
                self.phase = Phase::Discard;
                return;
            }

            let drawn = p.last_drawn_tile;
            let is_jishiyu = drawn == Some(tile);

            remove_tile(&mut p.hand, tile);
            // Convert Pon to KaKan
            if let Some(pos) = p.melds.iter().position(|m| matches!(m, MeldType::Pon(t) if *t == tile)) {
                p.melds[pos] = MeldType::KaKan(tile);
            }

            for i in 0..NUM_PLAYERS {
                if i != player_id && !self.players[i].has_won {
                    self.players[i].see_tile(tile);
                }
            }

            self.events.push(Event::KaKan { player: player_id, tile, is_jishiyu });

            // KaKan payment: 及时雨 only
            let mut jishiyu_paid = false;
            if is_jishiyu {
                jishiyu_paid = true;
                for i in 0..NUM_PLAYERS {
                    if i != player_id && !self.players[i].has_won {
                        self.players[i].score -= 1000;
                        self.players[player_id].score += 1000;
                        self.events.push(Event::KanPayment {
                            payer: i, receiver: player_id, amount: 1000,
                        });
                    }
                }
            }

            // Chankan: other players can ron the kakan tile
            let chankan_winners = self.check_chankan(player_id, tile);
            if !chankan_winners.is_empty() {
                // Revert kakan → restore to pon
                if let Some(pos) = self.players[player_id].melds.iter().position(
                    |m| matches!(m, MeldType::KaKan(t) if *t == tile)
                ) {
                    self.players[player_id].melds[pos] = MeldType::Pon(tile);
                }
                add_tile(&mut self.players[player_id].hand, tile);

                // Fix R11-M4: revert tiles_seen for non-winning opponents.
                // see_tile was called at kakan time (line 424), but chankan
                // reverts the kakan to pon, so the extra tile is not visible.
                for i in 0..NUM_PLAYERS {
                    if i != player_id && !self.players[i].has_won
                        && !chankan_winners.contains(&i)
                    {
                        self.players[i].unsee_tile(tile);
                    }
                }

                // Refund jishiyu payment if paid
                if jishiyu_paid {
                    for i in 0..NUM_PLAYERS {
                        if i != player_id && !self.players[i].has_won {
                            self.players[i].score += 1000;
                            self.players[player_id].score -= 1000;
                        }
                    }
                }

                for &winner in &chankan_winners {
                    self.events.push(Event::Ron { player: winner, from: player_id, tile });
                    self.process_chankan_win(winner, player_id, tile);
                }

                if self.win_count >= 3 {
                    self.phase = Phase::Scoring;
                } else {
                    // Kakan player has the tile restored to hand, needs to discard
                    self.current_player = player_id;
                    self.phase = Phase::Discard;
                }
                return;
            }
        }

        // Draw from back of wall (rinshan)
        if self.wall_remaining() > 0 {
            let new_tile = self.draw_from_back();
            let p = &mut self.players[player_id];
            add_tile(&mut p.hand, new_tile);
            p.last_drawn_tile = Some(new_tile);
            p.is_rinshan = true;
            p.see_tile(new_tile);

            // Clear 过手加番 on hand change (rinshan draw)
            p.furiten_passed_ron_fan = None;

            // Emit Draw event so the replay viewer shows the rinshan tile
            self.events.push(Event::Draw { player: player_id, tile: new_tile });

            self.current_player = player_id;
            self.phase = Phase::SelfCheck;
        } else {
            self.phase = Phase::Scoring;
        }
    }

    fn apply_discard(&mut self, player_id: usize, action: Action) {
        if player_id != self.current_player {
            return;
        }
        let tile = match action {
            Action::Discard(t) => {
                // Fix R11-M5: validate discard tile is legal before proceeding.
                // A malformed action (e.g. from a buggy action mask) would panic
                // in remove_tile. Fall back to first legal candidate if invalid.
                let candidates = self.players[player_id].discard_candidates();
                if candidates.contains(&t) {
                    t
                } else {
                    match candidates.first() {
                        Some(&fallback) => fallback,
                        None => return,
                    }
                }
            }
            _ => {
                // Non-discard action in discard phase (e.g. Pass from a confused model).
                // Force the first legal tile to keep the game moving.
                let candidates = self.players[player_id].discard_candidates();
                match candidates.first() {
                    Some(&t) => t,
                    None => return,
                }
            }
        };
        self.do_discard(player_id, tile);
    }

    fn do_discard(&mut self, player_id: usize, tile: Tile) {
        // Guard: validate tile is in hand before removing (prevents panic from stale action)
        if self.players[player_id].hand[tile as usize] == 0 {
            // Tile not in hand — fall back to first legal candidate
            let candidates = self.players[player_id].discard_candidates();
            if let Some(&fallback) = candidates.first() {
                if fallback != tile {
                    self.do_discard(player_id, fallback);
                }
            }
            return;
        }

        let p = &mut self.players[player_id];
        let was_rinshan = p.is_rinshan;
        let is_tsumogiri = p.last_drawn_tile == Some(tile);

        remove_tile(&mut p.hand, tile);
        p.discards.push(tile);
        p.tsumogiri.push(is_tsumogiri);
        p.last_drawn_tile = None;
        p.is_rinshan = false;

        for i in 0..NUM_PLAYERS {
            if i != player_id && !self.players[i].has_won {
                self.players[i].see_tile(tile);
            }
        }

        self.last_discard = Some((player_id, tile));
        self.last_discard_is_kan = was_rinshan;
        self.dahai_count += 1;

        self.events.push(Event::Discard {
            player: player_id,
            tile,
            is_tsumogiri,
        });

        // Set up reaction phase
        self.phase = Phase::Reaction;
        self.reactions = [None; NUM_PLAYERS];
        self.reaction_pending = [false; NUM_PLAYERS];

        let mut any_can_react = false;
        for i in 0..NUM_PLAYERS {
            if i == player_id || self.players[i].has_won { continue; }
            if self.get_reaction_actions(i).is_some() {
                self.reaction_pending[i] = true;
                any_can_react = true;
            }
        }

        if !any_can_react {
            self.resolve_reactions();
        }
    }

    fn apply_reaction(&mut self, player_id: usize, action: Action) {
        // Guard: only apply if this player actually has a pending reaction.
        // Without this, a spurious call (e.g. stale snapshot in Python loop) would
        // set reaction_pending[player_id]=false and trigger resolve_reactions() a
        // second time with stale data, advancing the game an extra turn.
        if !self.reaction_pending[player_id] {
            return;
        }
        self.reactions[player_id] = Some(action);
        self.reaction_pending[player_id] = false;

        // If action is Pass, record 过手加番 fan for this player
        if action == Action::Pass {
            if let Some((_, tile)) = self.last_discard {
                let p = &self.players[player_id];
                let shanten = calc_shanten(&p.hand, p.melds.len());
                if shanten == 0 {
                    let waits = waiting_tiles(&p.hand, p.melds.len());
                    if waits.contains(&tile) {
                        // Record fan for 过手加番
                        let ctx = self.make_win_context(player_id, tile, true);
                        if let Some(result) = calc_fan(&ctx) {
                            // Fix R11-H3: only raise threshold, never lower it.
                            // Non-agari passes (e.g. choosing pon/kan) could
                            // overwrite a higher fan threshold with a lower one,
                            // incorrectly relaxing the 过手加番 requirement.
                            let new_fan = result.fan;
                            match self.players[player_id].furiten_passed_ron_fan {
                                Some(existing) if existing >= new_fan => {
                                    // Keep the higher existing threshold
                                }
                                _ => {
                                    self.players[player_id].furiten_passed_ron_fan = Some(new_fan);
                                }
                            }
                        }
                    }
                }
            }
        }

        if self.reaction_pending.iter().all(|&p| !p) {
            self.resolve_reactions();
        }
    }

    fn resolve_reactions(&mut self) {
        let (discarder, tile) = match self.last_discard {
            Some(d) => d,
            None => { self.advance_to_next_draw(); return; }
        };

        // Priority: Ron > Pon/Kan > Pass
        let mut ron_players = Vec::new();
        let mut pon_player = None;
        let mut kan_player = None;

        for i in 0..NUM_PLAYERS {
            if let Some(action) = self.reactions[i] {
                match action {
                    Action::Agari => ron_players.push(i),
                    Action::Pon if pon_player.is_none() => pon_player = Some(i),
                    Action::Kan if kan_player.is_none() => kan_player = Some(i),
                    _ => {}
                }
            }
        }

        if !ron_players.is_empty() {
            for &winner in &ron_players {
                self.events.push(Event::Ron { player: winner, from: discarder, tile });
                self.process_win(winner, Some(discarder), tile);
            }
            if self.win_count >= 3 {
                self.phase = Phase::Scoring;
                return;
            }
            // Per rules: advance from the farthest winner's seat (in turn order from discarder)
            let last_winner = ron_players.iter()
                .max_by_key(|&&w| (w + NUM_PLAYERS - discarder) % NUM_PLAYERS)
                .copied().unwrap();
            self.current_player = last_winner;
            self.advance_to_next_draw();
        } else if let Some(kanner) = kan_player {
            // MinKan takes priority over Pon (higher fan potential)
            self.execute_minkan(kanner, discarder, tile);
        } else if let Some(ponner) = pon_player {
            self.execute_pon(ponner, discarder, tile);
        } else {
            self.advance_to_next_draw();
        }
    }

    fn execute_pon(&mut self, player_id: usize, from: usize, tile: Tile) {
        // Guard: need at least 2 copies in hand to pon
        if self.players[player_id].hand[tile as usize] < 2 {
            // Stale action: hand no longer has 2 copies. Treat as Pass.
            self.advance_to_next_draw();
            return;
        }
        let p = &mut self.players[player_id];
        remove_tile(&mut p.hand, tile);
        remove_tile(&mut p.hand, tile);
        p.melds.push(MeldType::Pon(tile));
        p.meld_from.push(Some(from)); // 碰的来源玩家
        p.furiten_passed_ron_fan = None;

        // The claimed tile is always the most recent discard — pop() is O(1)
        // and preserves the order of all earlier discards.
        self.players[from].discards.pop();
        self.players[from].tsumogiri.pop();

        for i in 0..NUM_PLAYERS {
            if i != player_id && !self.players[i].has_won {
                self.players[i].see_tile_n(tile, 2); // 2 tiles from hand
            }
        }

        self.events.push(Event::Pon { player: player_id, from, tile });
        self.current_player = player_id;
        self.phase = Phase::Discard;
    }

    fn execute_minkan(&mut self, player_id: usize, from: usize, tile: Tile) {
        // Guard: need at least 3 copies in hand to minkan
        if self.players[player_id].hand[tile as usize] < 3 {
            // Stale action: hand no longer has 3 copies. Treat as Pass.
            self.advance_to_next_draw();
            return;
        }
        let p = &mut self.players[player_id];
        remove_tile(&mut p.hand, tile);
        remove_tile(&mut p.hand, tile);
        remove_tile(&mut p.hand, tile);
        p.melds.push(MeldType::MinKan(tile));
        p.meld_from.push(Some(from)); // 明杠的来源玩家

        // The claimed tile is always the most recent discard — pop() is O(1).
        self.players[from].discards.pop();
        self.players[from].tsumogiri.pop();

        for i in 0..NUM_PLAYERS {
            if i != player_id && !self.players[i].has_won {
                self.players[i].see_tile_n(tile, 3); // 3 tiles from hand
            }
        }

        self.events.push(Event::MinKan { player: player_id, from, tile });

        // MinKan payment: discarder pays 2000
        self.players[from].score -= 2000;
        self.players[player_id].score += 2000;
        self.events.push(Event::KanPayment {
            payer: from, receiver: player_id, amount: 2000,
        });

        // Draw from back
        if self.wall_remaining() > 0 {
            let new_tile = self.draw_from_back();
            let p = &mut self.players[player_id];
            add_tile(&mut p.hand, new_tile);
            p.last_drawn_tile = Some(new_tile);
            p.is_rinshan = true;
            p.see_tile(new_tile);

            // Clear 过手加番 on hand change (minkan rinshan draw)
            p.furiten_passed_ron_fan = None;

            self.current_player = player_id;
            self.phase = Phase::SelfCheck;
        } else {
            self.phase = Phase::Scoring;
        }
    }

    fn process_win(&mut self, winner: usize, loser: Option<usize>, winning_tile: Tile) {
        let ctx = self.make_win_context(winner, winning_tile, loser.is_some());

        // Safety net: if calc_fan returns None (invalid win — hand not complete,
        // ding-que tiles remain, or wrong tile count), recover gracefully.
        // - Tsumo (loser=None): player has 14 tiles; go to Discard so they shed one.
        //   Do NOT call advance_to_next_draw (would leave 14-tile hand and draw for
        //   the next player, eventually causing a 15-tile hand).
        // - Ron (loser=Some): player has 13 tiles (winning tile only in copy);
        //   just return — resolve_reactions() handles advancement after all rons.
        let Some(result) = calc_fan(&ctx) else {
            if loser.is_none() {
                self.phase = Phase::Discard;
            }
            return;
        };
        let score = calc_score(result.fan);

        match loser {
            Some(from) => {
                self.players[from].score -= score;
                self.players[winner].score += score;
            }
            None => {
                for i in 0..NUM_PLAYERS {
                    if i != winner && !self.players[i].has_won {
                        self.players[i].score -= score;
                        self.players[winner].score += score;
                    }
                }
            }
        }

        self.players[winner].has_won = true;
        self.win_count += 1;

        if self.win_count >= 3 {
            self.phase = Phase::Scoring;
        } else if loser.is_none() {
            // Tsumo: advance immediately (no second caller will do it).
            self.advance_to_next_draw();
        }
        // Ron (loser.is_some()): resolve_reactions() calls advance_to_next_draw()
        // after all winners are collected, so we must NOT call it here — doing so
        // would cause a double-draw (the next player gets two tiles).
    }

    fn advance_to_next_draw(&mut self) {
        let mut next = (self.current_player + 1) % NUM_PLAYERS;
        for _ in 0..NUM_PLAYERS {
            if !self.players[next].has_won {
                break;
            }
            next = (next + 1) % NUM_PLAYERS;
        }
        if self.players[next].has_won {
            self.phase = Phase::Scoring;
            return;
        }

        if self.wall_remaining() == 0 {
            self.phase = Phase::Scoring;
            return;
        }

        // Draw tile
        let tile = self.draw_from_wall();
        let p = &mut self.players[next];
        add_tile(&mut p.hand, tile);
        p.last_drawn_tile = Some(tile);
        p.is_rinshan = false;
        p.see_tile(tile);
        p.furiten_passed_ron_fan = None;

        self.events.push(Event::Draw { player: next, tile });
        self.current_player = next;
        self.turn_count += 1;
        self.phase = Phase::SelfCheck;

        // Check if self-check has any special actions, otherwise go to discard
        if self.get_self_check_actions(next).is_none() {
            self.phase = Phase::Discard;
        }
    }

    fn make_win_context(&self, player_id: usize, winning_tile: Tile, is_ron: bool) -> WinContext {
        let p = &self.players[player_id];
        let mut tehai = p.hand;
        if is_ron {
            add_tile(&mut tehai, winning_tile);
        }
        WinContext {
            tehai,
            melds: p.melds.clone(),
            winning_tile,
            is_ron,
            ding_que: p.ding_que,
            is_after_kan: p.is_rinshan,
            is_kan_discard: self.last_discard_is_kan,
            is_chankan: false,
            is_haidi: self.wall_remaining() == 0,
            is_tianhu: !is_ron && self.dahai_count == 0 && player_id == self.dealer,
            // 地胡：闲家在第一巡自摸（非荣和、该玩家尚未打过牌、非庄家）
            // 额外要求：第一巡无人鸣牌（碰/杠），否则摸牌顺序被打断，不算地胡。
            is_dihu: !is_ron && p.discards.is_empty() && player_id != self.dealer
                && self.players.iter().all(|pl| pl.melds.is_empty()),
            exclude_gen_tile: None,
            fan_config: self.fan_config,  // Copy
        }
    }

    /// End-of-game scoring: 查花猪 + 查大叫
    pub fn finalize_scoring(&mut self) {
        // 查花猪: players who haven't completed ding que pay max hand to each tenpai player
        let mut hua_zhu = Vec::new(); // players with ding que tiles remaining
        let mut tenpai_players = Vec::new();

        for i in 0..NUM_PLAYERS {
            if self.players[i].has_won { continue; }
            if !self.players[i].ding_que_completed() {
                hua_zhu.push(i);
            } else {
                let shanten = calc_shanten(&self.players[i].hand, self.players[i].melds.len());
                if shanten == 0 {
                    tenpai_players.push(i);
                }
            }
        }

        // Pre-compute max score once per tenpai player (avoids O(n²) waiting_tiles calls)
        let max_scores: Vec<i32> = tenpai_players.iter()
            .map(|&tp| self.calc_max_hand_score(tp))
            .collect();

        // 花猪 pays max possible points to each tenpai player
        for &hz in &hua_zhu {
            for (j, &tp) in tenpai_players.iter().enumerate() {
                let max_score = max_scores[j];
                self.players[hz].score -= max_score;
                self.players[tp].score += max_score;
            }
        }

        // 查大叫: non-tenpai players (who completed ding que) pay tenpai players
        for i in 0..NUM_PLAYERS {
            if self.players[i].has_won { continue; }
            if hua_zhu.contains(&i) { continue; }
            if tenpai_players.contains(&i) { continue; }

            // This player is not tenpai and not hua zhu → pays tenpai
            for (j, &tp) in tenpai_players.iter().enumerate() {
                let max_score = max_scores[j];
                self.players[i].score -= max_score;
                self.players[tp].score += max_score;
            }
        }

        self.phase = Phase::Done;
        self.events.push(Event::GameEnd);
    }

    fn calc_max_hand_score(&self, player_id: usize) -> i32 {
        let p = &self.players[player_id];
        let waits = waiting_tiles(&p.hand, p.melds.len());
        let mut max_score = 0i32;

        for wt in waits {
            let mut h = p.hand;
            add_tile(&mut h, wt);
            let ctx = WinContext {
                tehai: h,
                melds: p.melds.clone(),
                winning_tile: wt,
                is_ron: false, // assume tsumo for max
                ding_que: p.ding_que,
                is_after_kan: false,
                is_kan_discard: false,
                is_chankan: false,
                is_haidi: false,
                is_tianhu: false,
                is_dihu: false,
                exclude_gen_tile: None,
                fan_config: self.fan_config,
            };
            if let Some(result) = calc_fan(&ctx) {
                let s = calc_score(result.fan);
                max_score = max_score.max(s);
            }
        }

        max_score
    }

    pub fn get_scores(&self) -> [i32; NUM_PLAYERS] {
        std::array::from_fn(|i| self.players[i].score)
    }

    pub fn get_rewards(&self, player_id: usize, prev_score: i32) -> f32 {
        let current = self.players[player_id].score;
        (current - prev_score) as f32 / REWARD_NORM as f32
    }

    fn check_chankan(&self, kakan_player: usize, tile: Tile) -> Vec<usize> {
        let mut winners = Vec::new();
        for i in 0..NUM_PLAYERS {
            if i == kakan_player || self.players[i].has_won { continue; }
            let p = &self.players[i];
            if !p.ding_que_completed() { continue; }

            let mut h = p.hand;
            add_tile(&mut h, tile);
            if !is_complete(&h, p.melds.len()) { continue; }

            // Always validate with calc_fan before adding to winners.
            // This prevents phantom Ron events when process_chankan_win would
            // fail validation (e.g. ding-que tiles remain, invalid division).
            let ctx = self.make_chankan_win_context(i, tile);
            let Some(result) = calc_fan(&ctx) else { continue; };

            // Blood mahjong has no furiten; only 过手加番 applies.
            if let Some(passed_fan) = p.furiten_passed_ron_fan {
                if result.fan > passed_fan {
                    winners.push(i);
                }
            } else {
                winners.push(i);
            }
        }
        winners
    }

    fn make_chankan_win_context(&self, player_id: usize, winning_tile: Tile) -> WinContext {
        let p = &self.players[player_id];
        let mut tehai = p.hand;
        add_tile(&mut tehai, winning_tile);
        WinContext {
            tehai,
            melds: p.melds.clone(),
            winning_tile,
            is_ron: true,
            ding_que: p.ding_que,
            is_after_kan: false,
            is_kan_discard: false,
            is_chankan: true,
            is_haidi: self.wall_remaining() == 0,
            is_tianhu: false,
            is_dihu: false,
            exclude_gen_tile: None,
            fan_config: self.fan_config,
        }
    }

    fn process_chankan_win(&mut self, winner: usize, loser: usize, winning_tile: Tile) {
        let ctx = self.make_chankan_win_context(winner, winning_tile);
        let Some(result) = calc_fan(&ctx) else { return; };
        let score = calc_score(result.fan);
        self.players[loser].score -= score;
        self.players[winner].score += score;

        self.players[winner].has_won = true;
        self.win_count += 1;
    }

    /// Public version of make_win_context for observation encoding.
    /// Fix R12-H1: provide both ron and tsumo variants so oracle can take max fan.
    pub fn make_win_context_for_obs(&self, player_id: usize, winning_tile: Tile) -> crate::algo::agari::WinContext {
        self.make_win_context(player_id, winning_tile, true)
    }

    pub fn make_win_context_for_obs_tsumo(&self, player_id: usize, winning_tile: Tile) -> crate::algo::agari::WinContext {
        self.make_win_context(player_id, winning_tile, false)
    }
}
