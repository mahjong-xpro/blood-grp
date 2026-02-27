"""Test full augmentation round-trip: forward + inverse."""

import pytest
from blood.env.augment import augment_action, SUIT_PERMUTATIONS


def test_augmentation_roundtrip():
    """Verify that forward + inverse augmentation is identity.
    
    Flow:
    1. Engine produces action in original space
    2. augment_action(action, perm) -> augmented action (model sees this)
    3. Model outputs augmented action
    4. inv_perm = tuple(perm.index(i) for i in range(3))
    5. augment_action(augmented_action, inv_perm) -> original action (engine receives this)
    """
    
    for perm in SUIT_PERMUTATIONS:
        # Compute inverse permutation
        inv_perm = tuple(perm.index(i) for i in range(3))
        
        # Test all actions
        for original_action in range(34):
            # Forward: engine -> model
            augmented = augment_action(original_action, perm)
            
            # Inverse: model -> engine
            recovered = augment_action(augmented, inv_perm)
            
            assert recovered == original_action, (
                f"Round-trip failed for action {original_action} with perm {perm}:\n"
                f"  original={original_action}\n"
                f"  augmented={augmented}\n"
                f"  recovered={recovered}\n"
                f"  inv_perm={inv_perm}"
            )


def test_inverse_permutation_correctness():
    """Verify inverse permutation calculation."""
    
    for perm in SUIT_PERMUTATIONS:
        inv_perm = tuple(perm.index(i) for i in range(3))
        
        # Check that perm[inv_perm[i]] == i for all i
        for i in range(3):
            assert perm[inv_perm[i]] == i, (
                f"Inverse permutation incorrect for perm {perm}:\n"
                f"  inv_perm={inv_perm}\n"
                f"  perm[inv_perm[{i}]] = perm[{inv_perm[i]}] = {perm[inv_perm[i]]} != {i}"
            )


def test_specific_case():
    """Test specific case that was failing."""
    
    # perm = (2, 0, 1): Man->Sou, Pin->Man, Sou->Pin
    perm = (2, 0, 1)
    inv_perm = tuple(perm.index(i) for i in range(3))
    
    print(f"\nperm = {perm}")
    print(f"inv_perm = {inv_perm}")
    
    # Test dingque actions
    for action in [31, 32, 33]:
        augmented = augment_action(action, perm)
        recovered = augment_action(augmented, inv_perm)
        
        print(f"\naction {action}:")
        print(f"  augmented = {augmented}")
        print(f"  recovered = {recovered}")
        
        assert recovered == action


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])