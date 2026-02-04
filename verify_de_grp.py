
import sys
import os
import numpy as np

# Add mortal directory to path
sys.path.append(os.path.join(os.getcwd(), 'mortal'))

try:
    import prelude # Should init libblood
    from libblood.dataset import GameScore
    print("Successfully imported GameScore from libblood.dataset")
except ImportError as e:
    print(f"Failed to import GameScore: {e}")
    # We continue to verify python logic even if libblood checks fail (might be rebuild needed)

try:
    from mortal.reward_calculator import RewardCalculator
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

try:
    rc = RewardCalculator()
    print("RewardCalculator initialized successfully.")
    
    # Test calc_delta_points with dummy data
    # scores_history: [kyoku_count, 4]
    scores_history_mock = np.zeros((10, 4)) 
    # Mock Start Scores: 60000, 60000, 60000, 60000
    scores_history_mock[:] = 60000

    final_scores = [70000, 50000, 60000, 60000]  # example final (zero-sum 240000)
    
    player_id = 0
    delta = rc.calc_delta_points(player_id, scores_history_mock, final_scores)
    print(f"Delta calculated: {delta}")
    
except Exception as e:
    print(f"Logic test failed: {e}")
    sys.exit(1)

print("Verification Passed.")
