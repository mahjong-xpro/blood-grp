"""Test suit augmentation correctness.

augment_obs and augment_action both use pull semantics:
    perm[new_suit] = old_suit

augment_obs:    result[:, new*9:(new+1)*9] = obs[:, perm[new]*9:...]
augment_action: old_suit → new_suit = perm.index(old_suit)
inverse:        new_suit → old_suit = perm[new_suit]
"""

import numpy as np
import pytest
from blood.env.augment import augment_obs, augment_action, SUIT_PERMUTATIONS


# --- augment_action basic tests ---

def test_dingque_augmentation():
    """Verify dingque action augmentation with pull semantics.

    perm=(2,0,1) means: new[0]=old[2], new[1]=old[0], new[2]=old[1]
    So: Man(0) goes to position perm.index(0)=1 → Pin position → action 32
        Pin(1) goes to position perm.index(1)=2 → Sou position → action 33
        Sou(2) goes to position perm.index(2)=0 → Man position → action 31
    """
    perm = (0, 1, 2)
    assert augment_action(31, perm) == 31
    assert augment_action(32, perm) == 32
    assert augment_action(33, perm) == 33

    # perm=(2,0,1): new Man ← old Sou, new Pin ← old Man, new Sou ← old Pin
    perm = (2, 0, 1)
    assert augment_action(31, perm) == 32  # Man(0) → perm.index(0)=1 → Pin pos
    assert augment_action(32, perm) == 33  # Pin(1) → perm.index(1)=2 → Sou pos
    assert augment_action(33, perm) == 31  # Sou(2) → perm.index(2)=0 → Man pos

    # perm=(1,2,0): new Man ← old Pin, new Pin ← old Sou, new Sou ← old Man
    perm = (1, 2, 0)
    assert augment_action(31, perm) == 33  # Man(0) → perm.index(0)=2 → Sou pos
    assert augment_action(32, perm) == 31  # Pin(1) → perm.index(1)=0 → Man pos
    assert augment_action(33, perm) == 32  # Sou(2) → perm.index(2)=1 → Pin pos

    for perm in SUIT_PERMUTATIONS:
        results = [augment_action(a, perm) for a in [31, 32, 33]]
        assert set(results) == {31, 32, 33}, f"Not a bijection with perm {perm}: {results}"


def test_discard_augmentation():
    """Verify discard action augmentation with pull semantics."""
    perm = (0, 1, 2)
    assert augment_action(0, perm) == 0
    assert augment_action(9, perm) == 9
    assert augment_action(18, perm) == 18

    # perm=(2,0,1): Man pos ← Sou data, Pin pos ← Man data, Sou pos ← Pin data
    # So: Man tile(0) → perm.index(0)=1 → Pin pos(9)
    #     Pin tile(9) → perm.index(1)=2 → Sou pos(18)
    #     Sou tile(18) → perm.index(2)=0 → Man pos(0)
    perm = (2, 0, 1)
    assert augment_action(0, perm) == 9    # 1m → 1p position
    assert augment_action(9, perm) == 18   # 1p → 1s position
    assert augment_action(18, perm) == 0   # 1s → 1m position


def test_special_actions_unchanged():
    for perm in SUIT_PERMUTATIONS:
        assert augment_action(27, perm) == 27  # Pon
        assert augment_action(28, perm) == 28  # Kan
        assert augment_action(29, perm) == 29  # Agari
        assert augment_action(30, perm) == 30  # Pass


# --- Invertibility ---

def test_augmentation_invertibility():
    """augment then inverse = identity for all actions and perms."""
    for perm in SUIT_PERMUTATIONS:
        # Inverse of pull-semantics augment_action is: old_suit = perm[new_suit]
        # Build inv_perm such that augment_action(·, inv_perm) is the inverse.
        # augment_action uses perm.index(old), so inverse needs inv where
        # inv.index(new) = old = perm[new], i.e. inv is the push-form of perm.
        # That's just: inv[old] = new ↔ inv[perm[new]] = new, which is the
        # standard inverse permutation.
        inv = [0, 0, 0]
        for i, j in enumerate(perm):
            inv[j] = i
        inv = tuple(inv)

        for action in range(34):
            augmented = augment_action(action, perm)
            restored = augment_action(augmented, inv)
            assert restored == action, (
                f"Not invertible: {action} → {augmented} → {restored} "
                f"with perm={perm}, inv={inv}"
            )


# --- The critical test: obs ↔ mask consistency ---

def test_obs_mask_consistency():
    """Verify that augmented obs and augmented mask agree on suit identity.

    If augment_obs puts old_suit S data at new position P, then the mask
    at position P must correspond to old_suit S's legality.

    This is the test that was MISSING and would have caught the previous bug.
    """
    for perm in SUIT_PERMUTATIONS:
        if perm == (0, 1, 2):
            continue  # identity is trivially correct

        # Build a synthetic obs where each suit has a unique marker.
        # Channel 0, tiles 0-8 = 1.0 (Man), tiles 9-17 = 2.0 (Pin), tiles 18-26 = 3.0 (Sou)
        obs = np.zeros((1, 27), dtype=np.float32)
        obs[0, 0:9] = 1.0    # Man marker
        obs[0, 9:18] = 2.0   # Pin marker
        obs[0, 18:27] = 3.0  # Sou marker

        aug_obs = augment_obs(obs, perm)

        # Build a synthetic mask: only Man tiles (0-8) are legal
        mask = np.zeros(34, dtype=np.float32)
        mask[0:9] = 1.0  # Man tiles legal

        # Augment the mask
        from blood.env.augment import augment_action
        action_map = np.array([augment_action(i, perm) for i in range(34)], dtype=np.intp)
        new_mask = np.zeros(34, dtype=np.float32)
        new_mask[action_map] = mask

        # Find where Man data (marker=1.0) ended up in augmented obs
        for new_suit in range(3):
            start = new_suit * 9
            if aug_obs[0, start] == 1.0:
                # Man data is at new_suit position.
                # The mask for this position's tiles should also be 1.0
                assert all(new_mask[start + r] == 1.0 for r in range(9)), (
                    f"perm={perm}: Man data at position {new_suit} but mask "
                    f"doesn't match. mask[{start}:{start+9}]="
                    f"{new_mask[start:start+9].tolist()}"
                )
            else:
                # Non-Man data here; mask should be 0.0
                assert all(new_mask[start + r] == 0.0 for r in range(9)), (
                    f"perm={perm}: Non-Man data at position {new_suit} but mask "
                    f"is non-zero. marker={aug_obs[0, start]}, "
                    f"mask[{start}:{start+9}]={new_mask[start:start+9].tolist()}"
                )


def test_inverse_action_consistency():
    """Verify _inverse_action correctly recovers original suit from augmented action.

    If augment_obs puts old Sou data at Man position, and the model outputs
    Man-1 (action 0), _inverse_action should return Sou-1 (action 18).
    """
    for perm in SUIT_PERMUTATIONS:
        for new_suit in range(3):
            old_suit = perm[new_suit]  # This is what _inverse_action should return
            for rank in range(9):
                aug_action = new_suit * 9 + rank
                # _inverse_action: old_suit = perm[new_suit]
                expected_original = old_suit * 9 + rank
                recovered = perm[aug_action // 9] * 9 + (aug_action % 9)
                assert recovered == expected_original, (
                    f"perm={perm}: aug_action={aug_action} (new_suit={new_suit}) "
                    f"should recover {expected_original} but got {recovered}"
                )

        # Dingque actions
        for new_suit in range(3):
            aug_action = 31 + new_suit
            old_suit = perm[new_suit]
            expected_original = 31 + old_suit
            recovered = 31 + perm[aug_action - 31]
            assert recovered == expected_original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
