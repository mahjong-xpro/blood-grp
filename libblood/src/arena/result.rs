use crate::mjai::{Event, EventExt};
use crate::rankings::Rankings;

use anyhow::Result;
use serde_json as json;

#[derive(Debug, Clone)]
pub struct KyokuResult {
    pub kyoku: u8,
    pub scores: [i32; 4],
    /// 和牌顺序（先和者在前）。用于同分排名。
    pub agari_order: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct GameResult {
    pub names: [String; 4],
    pub scores: [i32; 4],
    pub seed: (u64, u64),
    pub game_log: Vec<Vec<EventExt>>,
    /// Final hand tiles per player at game end, for display (e.g. 显示AI手牌).
    pub final_tehais: Option<Vec<Vec<String>>>,
    /// 和牌顺序（先和者在前），来自最后一局。用于同分排名。
    pub agari_order: Vec<u8>,
}

impl GameResult {
    #[inline]
    pub fn rankings(&self) -> Rankings {
        Rankings::new_with_agari_order(self.scores, Some(&self.agari_order))
    }

    pub fn dump_json_log(&self) -> Result<String> {
        let mut v = vec![];

        let start_game = Event::StartGame {
            names: self.names.clone(),
            seed: Some(self.seed),
        };
        json::to_writer(&mut v, &start_game)?;
        v.push(b'\n');

        for ev in self.game_log.iter().flatten() {
            json::to_writer(&mut v, ev)?;
            v.push(b'\n');
        }

        json::to_writer(&mut v, &Event::EndGame)?;
        v.push(b'\n');

        Ok(String::from_utf8(v)?)
    }
}
