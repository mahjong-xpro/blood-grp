use super::{Agent, BatchifiedAgent, InvisibleState};
use crate::mjai::{Event, EventExt, Suit};
use crate::state::PlayerState;

use anyhow::{Context, Result};

/// `Tsumogiri` always performs tsumogiri in all case and will not emit any
/// action other than discard. During ding_que phase, it auto-selects the suit
/// with the fewest tiles.
pub struct Tsumogiri(pub u8);

impl Tsumogiri {
    pub fn new_batched(player_ids: &[u8]) -> Result<BatchifiedAgent<Self>> {
        BatchifiedAgent::new(|id| Ok(Self(id)), player_ids)
    }
}

impl Agent for Tsumogiri {
    fn name(&self) -> String {
        "tsumogiri".to_owned()
    }

    fn react(
        &mut self,
        _: &[EventExt],
        state: &PlayerState,
        _: Option<InvisibleState>,
    ) -> Result<EventExt> {
        let cans = state.last_cans();
        let ev = if cans.can_ding_que {
            // 定缺阶段：选择手牌张数最少的花色
            let tehai = state.tehai();
            let man_count: u8 = tehai[0..9].iter().sum();
            let pin_count: u8 = tehai[9..18].iter().sum();
            let sou_count: u8 = tehai[18..27].iter().sum();
            let suit = if man_count <= pin_count && man_count <= sou_count {
                Suit::Man
            } else if pin_count <= sou_count {
                Suit::Pin
            } else {
                Suit::Sou
            };
            Event::DingQue {
                actor: self.0,
                suit,
            }
        } else if cans.can_discard {
            Event::Dahai {
                actor: self.0,
                pai: state.last_self_tsumo().context("last tsumo is empty")?,
                tsumogiri: true,
            }
        } else {
            Event::None
        };
        Ok(EventExt::no_meta(ev))
    }
}
