"""Suit permutation data augmentation.

6 permutations of (Man, Pin, Sou) applied at the game seed level
to ensure consistent augmentation across the entire episode.
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
    perm: tuple of 3 suit indices, e.g. (2, 0, 1)
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
    """Permute a discard action according to suit permutation.

    Actions 0-26 are tile indices; 27+ are non-tile actions.
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
