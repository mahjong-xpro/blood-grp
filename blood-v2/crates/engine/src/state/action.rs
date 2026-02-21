use crate::tile::{Tile, Suit};
use crate::consts::*;

/// Actions a player can take (34 total)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Discard(Tile),          // 0-26
    Pon,                    // 27
    Kan,                    // 28 (any type; tile chosen in kan-select if multiple)
    Agari,                  // 29 (tsumo or ron by context)
    Pass,                   // 30
    DingQue(Suit),          // 31-33
}

impl Action {
    pub fn to_index(self) -> usize {
        match self {
            Action::Discard(t) => t as usize,
            Action::Pon => 27,
            Action::Kan => 28,
            Action::Agari => 29,
            Action::Pass => 30,
            Action::DingQue(Suit::Man) => 31,
            Action::DingQue(Suit::Pin) => 32,
            Action::DingQue(Suit::Sou) => 33,
        }
    }

    pub fn from_index(idx: usize) -> Option<Action> {
        match idx {
            0..=26 => Some(Action::Discard(idx as Tile)),
            27 => Some(Action::Pon),
            28 => Some(Action::Kan),
            29 => Some(Action::Agari),
            30 => Some(Action::Pass),
            31 => Some(Action::DingQue(Suit::Man)),
            32 => Some(Action::DingQue(Suit::Pin)),
            33 => Some(Action::DingQue(Suit::Sou)),
            _ => None,
        }
    }
}

/// Legal actions available at a decision point
#[derive(Debug, Clone, Default)]
pub struct ActionCandidate {
    pub can_discard: bool,
    pub discard_tiles: Vec<Tile>,
    pub can_pon: bool,
    pub can_kan: bool,
    pub kan_tiles: Vec<Tile>,
    pub can_agari: bool,
    pub can_pass: bool,
    pub can_ding_que: bool,
    pub at_kan_select: bool,
}

impl ActionCandidate {
    pub fn to_mask(&self) -> [bool; ACTION_SPACE] {
        let mut mask = [false; ACTION_SPACE];

        if self.can_ding_que {
            mask[31] = true;
            mask[32] = true;
            mask[33] = true;
            return mask;
        }

        if self.at_kan_select {
            for &t in &self.kan_tiles {
                mask[t as usize] = true;
            }
            return mask;
        }

        if self.can_discard {
            for &t in &self.discard_tiles {
                mask[t as usize] = true;
            }
        }
        if self.can_pon { mask[27] = true; }
        if self.can_kan { mask[28] = true; }
        if self.can_agari { mask[29] = true; }
        if self.can_pass { mask[30] = true; }

        mask
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        let mut actions = Vec::new();

        if self.can_ding_que {
            actions.push(Action::DingQue(Suit::Man));
            actions.push(Action::DingQue(Suit::Pin));
            actions.push(Action::DingQue(Suit::Sou));
            return actions;
        }

        // KanSelect reuses Discard(tile) encoding: the tile index selects
        // which kan to execute. Action mask indices 0-26 map to tile types.
        if self.at_kan_select {
            for &t in &self.kan_tiles {
                actions.push(Action::Discard(t));
            }
            return actions;
        }

        if self.can_discard {
            for &t in &self.discard_tiles {
                actions.push(Action::Discard(t));
            }
        }
        if self.can_pon { actions.push(Action::Pon); }
        if self.can_kan { actions.push(Action::Kan); }
        if self.can_agari { actions.push(Action::Agari); }
        if self.can_pass { actions.push(Action::Pass); }

        actions
    }
}
