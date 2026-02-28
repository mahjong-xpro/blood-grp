"""Test full augmentation round-trip: forward + inverse.

After the fix, augment_action uses pull semantics (perm.index(old_suit)),
and _inverse_action uses perm[new_suit] directly (not augment_action with inv_perm).
"""

import pytest
from blood.env.augment import augment_action, SUIT_PERMUTATIONS


def test_augmentation_roundtrip():
    """Verify that augment_action then _inverse_action logic is identity.

    _inverse_action does: old_suit = perm[new_suit]
    augment_action does:  new_suit = perm.index(old_suit)

    Compose: old → new=perm.index(old) → recovered=perm[new]=perm[perm.index(old)]=old ✓
    """
    for perm in SUIT_PERMUTATIONS:
        for original_action in range(34):
            # Forward: augment_action
            augmented = augment_action(original_action, perm)

            # Inverse: perm[new_suit] (matching blood_env._inverse_action)
            if augmented >= 27:
                if 31 <= augmented <= 33:
                    new_suit = augmented - 31
                    recovered = 31 + perm[new_suit]
                else:
                    recovered = augmented
            else:
                new_suit = augmented // 9
                rank = augmented % 9
                recovered = perm[new_suit] * 9 + rank

            assert recovered == original_action, (
                f"Round-trip failed for action {original_action} with perm {perm}:\n"
                f"  augmented={augmented}, recovered={recovered}"
            )


def test_inverse_permutation_correctness():
    """Verify perm[perm.index(i)] == i for all i (mathematical identity)."""
    for perm in SUIT_PERMUTATIONS:
        for i in range(3):
            assert perm[perm.index(i)] == i, (
                f"perm={perm}: perm[perm.index({i})] = {perm[perm.index(i)]} != {i}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
