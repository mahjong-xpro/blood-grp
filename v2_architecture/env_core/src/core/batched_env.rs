//! Batched Environment interface for Ray/Python compatibility

use crate::core::state::GameState;

/// A vectorized environment manager handling multiple games simultaneously.
pub struct BatchedEnv {
    pub num_envs: usize,
    pub states: Vec<GameState>,
}

impl BatchedEnv {
    /// Creates a new Batched Environment with the specified capacity
    pub fn new(num_envs: usize) -> Self {
        let mut states = Vec::with_capacity(num_envs);
        for _ in 0..num_envs {
            states.push(GameState::new());
        }
        Self { num_envs, states }
    }

    /// Resets specific environment indices.
    /// If indices is empty, resets all environments.
    pub fn reset(&mut self, indices: Option<&[usize]>) {
        if let Some(idx_list) = indices {
            for &i in idx_list {
                if i < self.num_envs {
                    self.states[i].reset();
                }
            }
        } else {
            for state in self.states.iter_mut() {
                state.reset();
            }
        }
    }

    /// Performs a step for all active environments given their actions.
    /// This is currently a mocked structure returns an array of done statuses.
    /// In a fully integrated system, this returns stacked Numpy Arrays via PyO3.
    pub fn step(&mut self, actions: &[i32]) -> Vec<bool> {
        let mut dones = Vec::with_capacity(self.num_envs);
        
        for (i, state) in self.states.iter_mut().enumerate() {
            if state.is_done {
                dones.push(true);
                continue;
            }
            
            // Extract action
            let action = if i < actions.len() { actions[i] } else { -1 };
            
            // Apply DingQue
            // Actions 0-33: Discard
            // Actions 34: DingQue(Man), 35: DingQue(Pin), 36: DingQue(Sou)
            // Note: This is a simplified mapping for demonstration
            if action >= 34 && action <= 36 {
                let p = &mut state.players[state.current_player];
                p.missing_suit = Some((action - 34) as u8);
            } else if action >= 0 && action <= 26 {
                // Apply discard
                let tile = action as u8;
                let p = &mut state.players[state.current_player];
                if p.hand[tile as usize] > 0 {
                    p.hand[tile as usize] -= 1;
                    
                    // Proceed turn
                    state.current_player = (state.current_player + 1) % 4;
                    // Draw next tile
                    if let Some(drawn) = state.wall.pop() {
                        state.players[state.current_player].hand[drawn as usize] += 1;
                    } else {
                        // Wall empty
                        state.is_done = true;
                    }
                }
            }

            state.turn_count += 1;
            dones.push(state.is_done);
        }
        
        dones
    }
}
