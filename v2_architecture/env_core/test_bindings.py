import sys
import os

# Help python find the compiled rust lib
# Cargo puts it in target/release/libenv_core.dylib
# We need to symlink or rename it to env_core.so for Python to recognize it
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "target/release"))
sys.path.append(lib_path)

try:
    import env_core
    print("Successfully imported env_core Rust extension!")
    
    # Initialize Batched Environment with 4 games
    num_envs = 4
    env = env_core.PyBatchedEnv(num_envs)
    print(f"Created PyBatchedEnv with {env.num_envs()} environments.")
    
    # Reset
    print("Resetting environments...")
    env.reset(None)
    
    # Take a few steps
    print("Taking steps...")
    actions = [0, 1, 2, 3] # Dummy actions
    for step in range(3):
        dones = env.step(actions)
        print(f"  Step {step+1}: Dones={dones}")
        
    print("Test passed successfully!")
    
except ImportError as e:
    print(f"Failed to import env_core: {e}")
    # Print directory contents to help debug
    if os.path.exists(lib_path):
        print(f"Contents of {lib_path}:")
        print(os.listdir(lib_path))
