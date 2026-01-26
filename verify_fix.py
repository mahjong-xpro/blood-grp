import blood
import json
import sys

print(f"Checking blood module...")
try:
    shape = blood.consts.obs_shape(4)
    print(f"blood.consts.obs_shape(4): {shape}")
    if shape != (478, 27):
        print(f"FAILURE: Expected (478, 27), got {shape}")
        sys.exit(1)
    else:
        print("SUCCESS: obs_shape matches expected value.")
except Exception as e:
    print(f"Error checking obs_shape: {e}")
    sys.exit(1)

# Minimal log to trigger encoding
# We need enough events to trigger at least one encode_obs call (e.g. discard)
dummy_log = [
    {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
    {"type": "start_kyoku", "kyoku": 1, "oya": 0, "scores": [25000, 25000, 25000, 25000], 
     "tehais": [["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"], 
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"], 
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"], 
                ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]]},
    {"type": "tsumo", "actor": 0, "pai": "5p"},
    {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
    {"type": "end_kyoku"},
    {"type": "end_game"}
]

# Convert to string format expected by load_log (one json per line)
log_str = "\n".join(json.dumps(ev) for ev in dummy_log)

print("Testing GameplayLoader...")
try:
    loader = blood.dataset.GameplayLoader(4) # Version 4
    
    games = loader.load_log(log_str)
    print(f"Loaded {len(games)} gameplay objects")
    
    for i, game in enumerate(games):
        print(f"Checking game {i}...")
        # Access properties to trigger lazy evaluation or just check pre-computed
        # obs is Vec<Array2<f32>>, accessible via take_obs()
        obs_list = game.take_obs()
        print(f"  Obs count: {len(obs_list)}")
        if len(obs_list) > 0:
            print(f"  Obs shape: {obs_list[0].shape}")
            if obs_list[0].shape != (478, 27):
                 print(f"FAILURE: Obs shape {obs_list[0].shape} mismatch!")
                 sys.exit(1)
        
    print("SUCCESS: Encoding finished without panic and shapes match.")

except Exception as e:
    print(f"Error during loading/encoding: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
