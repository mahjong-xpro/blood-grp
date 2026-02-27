"""Test suit augmentation fix for dingque actions."""

import pytest
from blood.env.augment import augment_action, SUIT_PERMUTATIONS


def test_dingque_augmentation():
    """Verify dingque action augmentation works correctly.
    
    Bug: Original code used perm.index(old_suit) which is reverse mapping.
    Fix: Should use perm[old_suit] for forward mapping.
    
    Example with perm=(2,0,1):
    - Man (suit 0) -> Sou (suit 2): action 31 -> 33
    - Pin (suit 1) -> Man (suit 0): action 32 -> 31
    - Sou (suit 2) -> Pin (suit 1): action 33 -> 32
    """
    
    # Test identity permutation
    perm = (0, 1, 2)
    assert augment_action(31, perm) == 31  # Man -> Man
    assert augment_action(32, perm) == 32  # Pin -> Pin
    assert augment_action(33, perm) == 33  # Sou -> Sou
    
    # Test permutation (2, 0, 1): Man->Sou, Pin->Man, Sou->Pin
    perm = (2, 0, 1)
    assert augment_action(31, perm) == 33  # Man -> Sou
    assert augment_action(32, perm) == 31  # Pin -> Man
    assert augment_action(33, perm) == 32  # Sou -> Pin
    
    # Test permutation (1, 2, 0): Man->Pin, Pin->Sou, Sou->Man
    perm = (1, 2, 0)
    assert augment_action(31, perm) == 32  # Man -> Pin
    assert augment_action(32, perm) == 33  # Pin -> Sou
    assert augment_action(33, perm) == 31  # Sou -> Man
    
    # Test all 6 permutations
    for perm in SUIT_PERMUTATIONS:
        # Each dingque action should map to a valid dingque action
        for action in [31, 32, 33]:
            result = augment_action(action, perm)
            assert 31 <= result <= 33, f"Invalid result {result} for action {action} with perm {perm}"
        
        # All three dingque actions should map to different actions
        results = [augment_action(a, perm) for a in [31, 32, 33]]
        assert len(set(results)) == 3, f"Duplicate mapping with perm {perm}: {results}"
        assert set(results) == {31, 32, 33}, f"Invalid mapping with perm {perm}: {results}"


def test_discard_augmentation():
    """Verify discard action augmentation works correctly."""
    
    # Test identity
    perm = (0, 1, 2)
    assert augment_action(0, perm) == 0    # 1m -> 1m
    assert augment_action(9, perm) == 9    # 1p -> 1p
    assert augment_action(18, perm) == 18  # 1s -> 1s
    
    # Test permutation (2, 0, 1): Man->Sou, Pin->Man, Sou->Pin
    perm = (2, 0, 1)
    assert augment_action(0, perm) == 18   # 1m -> 1s
    assert augment_action(9, perm) == 0    # 1p -> 1m
    assert augment_action(18, perm) == 9   # 1s -> 1p
    
    # Test permutation (1, 2, 0): Man->Pin, Pin->Sou, Sou->Man
    perm = (1, 2, 0)
    assert augment_action(0, perm) == 9    # 1m -> 1p
    assert augment_action(9, perm) == 18   # 1p -> 1s
    assert augment_action(18, perm) == 0   # 1s -> 1m


def test_special_actions_unchanged():
    """Verify special actions (27-30) are not affected by augmentation."""
    
    for perm in SUIT_PERMUTATIONS:
        assert augment_action(27, perm) == 27  # Pon
        assert augment_action(28, perm) == 28  # Kan
        assert augment_action(29, perm) == 29  # Agari
        assert augment_action(30, perm) == 30  # Pass


def test_augmentation_invertibility():
    """Verify augmentation is invertible: augment(augment(a, p), p_inv) == a"""
    
    # For each permutation, find its inverse
    for perm in SUIT_PERMUTATIONS:
        # Inverse permutation: if perm[i] = j, then inv[j] = i
        inv = [0, 0, 0]
        for i, j in enumerate(perm):
            inv[j] = i
        inv = tuple(inv)
        
        # Test dingque actions
        for action in [31, 32, 33]:
            augmented = augment_action(action, perm)
            restored = augment_action(augmented, inv)
            assert restored == action, f"Not invertible: {action} -> {augmented} -> {restored} with perm {perm}"
        
        # Test discard actions
        for action in range(27):
            augmented = augment_action(action, perm)
            restored = augment_action(augmented, inv)
            assert restored == action, f"Not invertible: {action} -> {augmented} -> {restored} with perm {perm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])