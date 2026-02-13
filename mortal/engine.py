import json
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *

class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        name = 'NoName',
        boltzmann_epsilon = 0,
        boltzmann_temp = 1,
        top_p = 1,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        # Optimization: If input is already a list, stack it. If it's already an array/tensor, directly convert.
        if isinstance(obs, list):
            obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        else:
            obs = torch.as_tensor(obs, device=self.device)

        if isinstance(masks, list):
            masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        else:
            masks = torch.as_tensor(masks, device=self.device)
        masks = masks.to(torch.bool)

        if invisible_obs is not None:
            if isinstance(invisible_obs, list):
                invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
            else:
                invisible_obs = torch.as_tensor(invisible_obs, device=self.device)
        
        batch_size = obs.shape[0]
        if masks.shape[0] != batch_size:
            raise ValueError(f"batch size mismatch: obs.shape[0]={batch_size}, masks.shape[0]={masks.shape[0]}")

        valid_counts = masks.sum(-1)
        if (valid_counts == 0).any():
            bad = (valid_counts == 0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"invalid action mask: no valid actions for batch indices {bad}")

        if self.version == 1:
            mu, logsig = self.brain(obs, invisible_obs)
            if self.stochastic_latent:
                latent = Normal(mu, logsig.exp() + 1e-6).sample()
            else:
                latent = mu
            q_out = self.dqn(latent, masks)
        elif self.version in (2, 3, 4):
            # FIX: oracle 模式下必须传入 invisible_obs，否则 Brain.forward() 会 assert 失败。
            # 非 oracle 模式 invisible_obs 为 None，Brain.forward() 会忽略它。
            phi = self.brain(obs, invisible_obs)
            q_out = self.dqn(phi, masks)

        # AMP (enable_amp=True) 下 DQN 输出为 float16（范围 ±65504）。
        # 除以 boltzmann_temp（如 0.18 ≈ ×5.56）后，|Q| > 11780 的合法位置
        # 会溢出为 ±inf，导致 Categorical 崩溃。提升到 float32 消除溢出。
        q_out = q_out.float()

        if self.boltzmann_epsilon > 0:
            if self.boltzmann_temp <= 0:
                raise ValueError(f"boltzmann_temp must be > 0, got {self.boltzmann_temp}")
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            
            # 定缺动作 (31=万, 32=饼, 33=条) 强制使用贪婪选择，跳过探索
            # 只有当可用动作仅限于定缺时才跳过探索
            ding_que_only = (
                masks[:, 31] | masks[:, 32] | masks[:, 33]
            ) & (masks[:, :31].sum(-1) == 0) & (masks[:, 34:].sum(-1) == 0)
            is_greedy = is_greedy | ding_que_only  # 定缺时强制贪婪
            
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int], final_tehais=None):
        pass
