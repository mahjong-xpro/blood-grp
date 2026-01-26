import blood
import json
import sys
import numpy as np

# Log where kyoku = 7 (South 3). 
# Encode logic: 
# kyoku one-hot writes to idx + (7-1) = idx + 6.
# Block size is 4. Padding is 2. Total 6.
# So idx + 6 is the START of the NEXT block (Ding Que).
# Sending kyoku=7 means `state.kyoku` = 6. 
# One-hot: fill(idx + 6).
# Next block (Ding Que) starts at idx + 4 + 2 = idx + 6.
# So `kyoku=7` should simulate a "Man" suit DingQue selection if collision occurs.

dummy_log = [
    {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
    {"type": "start_kyoku", 
     "kyoku": 7,  # South 3
     "oya": 0, "scores": [25000, 25000, 25000, 25000], 
     "tehais": [["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"], 
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"], 
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"], 
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]]},
    {"type": "end_kyoku"},
    {"type": "end_game"}
]

log_str = "\n".join(json.dumps(ev) for ev in dummy_log)

print("Checking for Kyoku Collision...")
loader = blood.dataset.GameplayLoader(4)
games = loader.load_log(log_str)
game = games[0]
obs_list = game.take_obs()
obs = obs_list[0] # (478, 27)

# We need to find the index of Ding Que.
# It's hard to calculate exact index dynamically, but we can search for "Man" suit DingQue signal.
# Man suit DingQue is encoded as `fill(base, 1.)`. (all 27 cols? No, `fill` fills the row).
# So we look for a row that is all 1s.
# "kyoku" one-hot also uses `fill`. 
# So if collision happens, we will see a row of 1s where DingQue should be.
# BUT, we didn't set DingQue in the log.
# So normally, DingQue rows should be 0.
# If we find a row of 1s in the DingQue region, it's a collision.

# Let's locate the DingQue region approximately.
# Previous features:
# Tehai: 4
# Scores: 4
# Rank: 4
# Kyoku 1: 4
# Padding: 2
# Ding Que starts at roughly index 18?
# Let's verify exact offset.
# Tehai: 4.
# Scores: 4? No.
# Scores logic: 
#   Loop 4 scores. Each has 1 normalized + (V4) 1 rescaled? No.
#   V4: 
#     v = norm; idx+=1.
#     match V4: v=norm; idx+=1.
#   So 2 channels per score? 
#   Wait, let's re-read obs_repr.rs.

# Lines 141-158:
# for &score in &state.scores:
#    idx += 1 (raw norm)
#    match version 4:
#       idx += 1 (clamped norm)
# So 2 * 4 = 8 channels for scores.

# Rank: 4. (idx + 4)
# Kyoku 1: 4. (idx + 4)
# Padding: 2. (idx + 2)

# Total before DingQue: 4 (Tehai) + 8 (Scores) + 4 (Rank) + 4 (Kyoku) + 2 (Pad) = 22.
# DingQue starts at index 22.
# Kyoku=7 -> n=6. Writes to StartOfKyoku + 6.
# StartOfKyoku = 4+8+4 = 16.
# 16 + 6 = 22.
# So it writes to row 22.
# Row 22 is EXCACTLY the start of DingQue.

print(f"Checking row 22 (Expected DingQue 'Man' slot)...")
row_22 = obs[22]
print(f"Row 22 sum: {np.sum(row_22)}")
print(f"Row 22 mean: {np.mean(row_22)}")

if np.isclose(np.mean(row_22), 1.0):
    print("FAILURE: Collision Detected! Row 22 is filled (Kyoku leaking into DingQue).")
    sys.exit(1)
elif np.isclose(np.mean(row_22), 0.0):
    print("SUCCESS: Row 22 is empty. No collision.")
else:
    print(f"WARNING: Row 22 has unexpected value: {row_22}")

