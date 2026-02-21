use engine::tile::Suit;
use engine::hand::*;
use engine::state::board::{BoardState, Phase};
use engine::state::action::Action;
use engine::algo::shanten::calc_shanten;

#[derive(Debug, Clone)]
pub enum OpponentPolicy {
    RuleBot,
    Random(fastrand::Rng),
    External,
}

impl OpponentPolicy {
    pub fn choose_ding_que(&self, board: &BoardState, player_id: usize) -> Action {
        let p = &board.players[player_id];
        let mut best_suit = Suit::Man;
        let mut min_count = u8::MAX;
        for suit in Suit::all() {
            let count = suit_tile_count(&p.hand, suit);
            if count < min_count {
                min_count = count;
                best_suit = suit;
            }
        }
        Action::DingQue(best_suit)
    }

    pub fn choose_action(&mut self, board: &BoardState, player_id: usize) -> Action {
        match board.phase {
            Phase::SelfCheck => {
                if let Some(ac) = board.get_decision_request(player_id) {
                    if ac.can_agari {
                        return Action::Agari;
                    }
                    if ac.can_kan && matches!(self, OpponentPolicy::RuleBot) {
                        let p = &board.players[player_id];
                        let ankan = p.can_ankan_tiles();
                        if !ankan.is_empty() {
                            return Action::Kan;
                        }
                        let s_before = calc_shanten(&p.hand, p.melds.len());
                        if s_before > 0 {
                            return Action::Kan;
                        }
                    }
                }
                Action::Pass
            }
            Phase::KanSelect => {
                if let Some(ac) = board.get_decision_request(player_id) {
                    if !ac.kan_tiles.is_empty() {
                        return Action::Discard(ac.kan_tiles[0]);
                    }
                }
                Action::Pass
            }
            Phase::Discard => {
                self.choose_discard(board, player_id)
            }
            _ => Action::Pass,
        }
    }

    pub fn choose_reaction(&mut self, board: &BoardState, player_id: usize) -> Action {
        if let Some(ac) = board.get_decision_request(player_id) {
            if ac.can_agari {
                return Action::Agari;
            }
            match self {
                OpponentPolicy::RuleBot => {
                    if ac.can_pon {
                        let p = &board.players[player_id];
                        let s_before = calc_shanten(&p.hand, p.melds.len());
                        let mut h_after = p.hand;
                        if let Some((_, tile)) = board.last_discard {
                            remove_tile(&mut h_after, tile);
                            remove_tile(&mut h_after, tile);
                            let s_after = calc_shanten(&h_after, p.melds.len() + 1);
                            if s_after < s_before {
                                return Action::Pon;
                            }
                        }
                    }
                }
                OpponentPolicy::Random(rng) => {
                    let mut choices = vec![Action::Pass];
                    if ac.can_pon { choices.push(Action::Pon); }
                    if ac.can_kan { choices.push(Action::Kan); }
                    let idx = rng.usize(..choices.len());
                    return choices[idx];
                }
                OpponentPolicy::External => {}
            }
        }
        Action::Pass
    }

    fn choose_discard(&mut self, board: &BoardState, player_id: usize) -> Action {
        let p = &board.players[player_id];
        let candidates = p.discard_candidates();

        if candidates.is_empty() {
            return Action::Pass;
        }

        match self {
            OpponentPolicy::External => Action::Pass,
            OpponentPolicy::Random(rng) => {
                let idx = rng.usize(..candidates.len());
                Action::Discard(candidates[idx])
            }
            OpponentPolicy::RuleBot => {
                let mut best_tile = candidates[0];
                let mut best_shanten = 99i8;

                for &t in &candidates {
                    let mut h = p.hand;
                    remove_tile(&mut h, t);
                    let s = calc_shanten(&h, p.melds.len());
                    if s < best_shanten {
                        best_shanten = s;
                        best_tile = t;
                    }
                }
                Action::Discard(best_tile)
            }
        }
    }
}
