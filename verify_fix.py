import json
import os
import sys

# Ensure we can import the project-local loader
sys.path.append(os.path.join(os.getcwd(), "mortal"))

print("Initializing libblood...")
try:
    import prelude  # loads libblood via libblood_loader on macOS
    import libblood as blood
except Exception as e:
    raise SystemExit(
        "Failed to import libblood. Build it first:\n"
        "  cargo build -p libblood --release\n"
        f"Original error: {e}"
    )

print("Checking obs_shape(4)...")
shape = blood.consts.obs_shape(4)
print(f"libblood.consts.obs_shape(4): {shape}")
if shape != (461, 27):
    raise SystemExit(f"FAILURE: Expected (461, 27), got {shape}")
print("SUCCESS: obs_shape matches expected value.")

# Minimal log to trigger encoding (must include DingQue stage for Bloody Battle logs)
dummy_log = [
    {"type": "start_game", "names": ["p0", "p1", "p2", "p3"], "seed": [1, 1]},
    {
        "type": "start_kyoku",
        "kyoku": 1,
        "oya": 0,
        "scores": [60000, 60000, 60000, 60000],
        "tehais": [
            ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
            ["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","4s"],
            ["1s","2s","3s","4s","5s","6s","7s","8s","9s","1m","2m","3m","4m"],
            ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],
        ],
    },
    {"type": "ding_que", "actor": 0, "suit": "man"},
    {"type": "ding_que", "actor": 1, "suit": "pin"},
    {"type": "ding_que", "actor": 2, "suit": "sou"},
    {"type": "ding_que", "actor": 3, "suit": "man"},
    {"type": "tsumo", "actor": 0, "pai": "5p"},
    {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
    {"type": "end_kyoku"},
    {"type": "end_game"},
]

log_str = "\n".join(json.dumps(ev) for ev in dummy_log)

print("Testing GameplayLoader encoding...")
loader = blood.dataset.GameplayLoader(version=4, oracle=False)
games = loader.load_log(log_str)
print(f"Loaded {len(games)} gameplay objects")

for i, game in enumerate(games):
    obs_list = game.take_obs()
    print(f"game[{i}] obs_count={len(obs_list)}")
    if len(obs_list) > 0:
        got = obs_list[0].shape
        print(f"game[{i}] obs_shape={got}")
        if got != (461, 27):
            raise SystemExit(f"FAILURE: Obs shape {got} mismatch (expected (461, 27))")

print("SUCCESS: Encoding finished without panic and shapes match.")
