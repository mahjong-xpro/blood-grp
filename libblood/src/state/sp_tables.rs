use crate::algo::sp::Candidate;

#[derive(Clone)]
pub struct SinglePlayerTables {
    pub max_ev_table: Vec<Candidate>,
}
