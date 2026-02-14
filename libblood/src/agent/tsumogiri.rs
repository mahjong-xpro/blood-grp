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
            let suits = [Suit::Man, Suit::Pin, Suit::Sou];
            let suit = *suits.iter().min_by_key(|&&s| {
                let (start, end) = crate::ding_que::suit_range(s);
                tehai[start..end].iter().sum::<u8>()
            }).unwrap();
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
