import blood
import sys

print(f"Checking blood module for Oracle Shape...")
try:
    shape = blood.consts.oracle_obs_shape(4)
    print(f"blood.consts.oracle_obs_shape(4): {shape}")
    if shape != (118, 27):
        print(f"FAILURE: Expected (118, 27), got {shape}")
        sys.exit(1)
    else:
        print("SUCCESS: oracle_obs_shape matches expected value (118, 27).")
except Exception as e:
    print(f"Error checking oracle_obs_shape: {e}")
    sys.exit(1)
