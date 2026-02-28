"""Suit permutation data augmentation.

6 permutations of (Man, Pin, Sou) applied at the game seed level
to ensure consistent augmentation across the entire episode.

Permutation semantics (pull):
    perm[new_suit] = old_suit
    e.g. perm=(2,0,1) means: new Man position ← old Sou data,
                              new Pin position ← old Man data,
                              new Sou position ← old Pin data.

Both augment_obs and augment_action MUST use the same pull semantics
so that observation data and action legality at each suit position
refer to the same original suit.
"""

import numpy as np

SUIT_PERMUTATIONS = [
    (0, 1, 2),  # identity
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]


def augment_obs(obs, perm):
    """Permute observation channels according to suit permutation.

    obs: (C, 27) array
    perm: pull mapping — perm[new_suit] = old_suit
    """
    c, w = obs.shape
    assert w == 27
    result = np.zeros_like(obs)
    for new_suit, old_suit in enumerate(perm):
        src_start = old_suit * 9
        dst_start = new_suit * 9
        result[:, dst_start:dst_start + 9] = obs[:, src_start:src_start + 9]
    return result


def augment_action(action: int, perm) -> int:
    """Permute an action according to suit permutation (pull semantics).

    Actions 0-26 are tile indices; 27+ are non-tile actions.

    perm uses pull semantics: perm[new_suit] = old_suit.
    Given an original action in old_suit, we find the new_suit position
    where that old_suit's data now lives: new_suit = perm.index(old_suit).

    Example: perm=(2,0,1), action=18 (Sou-1, old_suit=2)
    - augment_obs puts Sou data at Man position (perm[0]=2)
    - So Sou-1 action should map to Man-1: perm.index(2)=0 → action 0
    """
    if action >= 27:
        if 31 <= action <= 33:
            old_suit = action - 31
            new_suit = perm.index(old_suit)
            return 31 + new_suit
        return action

    old_suit = action // 9
    rank = action % 9
    new_suit = perm.index(old_suit)
    return new_suit * 9 + rank
