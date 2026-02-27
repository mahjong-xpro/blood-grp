#!/bin/bash
# Complete DingQue Bug Fix Script
# This script implements all necessary fixes and recompiles the Rust engine

set -e  # Exit on error

echo "=========================================="
echo "DingQue Bug Complete Fix"
echo "=========================================="
echo ""

# Step 1: Verify we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "ERROR: Must run from blood-v2 directory"
    exit 1
fi

echo "Step 1: Checking Rust toolchain..."
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust/Cargo not found. Install from https://rustup.rs/"
    exit 1
fi
echo "✓ Rust toolchain found"
echo ""

echo "Step 2: Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi
if ! python3 -c "import maturin" 2>/dev/null; then
    echo "ERROR: maturin not installed. Run: pip install maturin"
    exit 1
fi
echo "✓ Python environment ready"
echo ""

echo "Step 3: Verifying observation fix is in place..."
if grep -q "for suit in 0..3" crates/engine/src/obs/student.rs; then
    echo "✓ Observation encoding fix found in student.rs"
else
    echo "ERROR: Observation fix not found in student.rs"
    echo "Please ensure the fix at line 94-113 is applied"
    exit 1
fi
echo ""

echo "Step 4: Verifying augmentation fix is in place..."
if grep -q "perm\[old_suit\]" python/blood/env/augment.py; then
    echo "✓ Augmentation mapping fix found in augment.py"
else
    echo "ERROR: Augmentation fix not found in augment.py"
    echo "Please ensure line 43 uses perm[old_suit] not perm.index(old_suit)"
    exit 1
fi
echo ""

echo "Step 5: Compiling Rust engine with fixes..."
echo "This may take 2-5 minutes..."
maturin develop --release
if [ $? -eq 0 ]; then
    echo "✓ Rust engine compiled successfully"
else
    echo "ERROR: Compilation failed"
    exit 1
fi
echo ""

echo "Step 6: Verifying blood_engine module is installed..."
if python3 -c "import blood_engine" 2>/dev/null; then
    echo "✓ blood_engine module imported successfully"
else
    echo "ERROR: blood_engine module not found after compilation"
    exit 1
fi
echo ""

echo "Step 7: Testing observation encoding..."
python3 << 'EOF'
import blood_engine
import numpy as np

# Create a test environment
env = blood_engine.BloodMahjongEnv(12345, "rulebot", 100000)
env.reset(12345)

# Get observation during DingQue phase
obs_dict = env.get_player_obs(0)
obs = np.array(obs_dict["obs"], dtype=np.float32)
obs_2d = obs.reshape(473, 27)

# Check Section 3 (channels 18-20) - should NOT be all zeros
section3 = obs_2d[18:21, :]
section3_sum = section3.sum()

print(f"Section 3 sum: {section3_sum}")
if section3_sum > 0:
    print("✓ Observation encoding is working (Section 3 has data)")
else:
    print("✗ WARNING: Section 3 is still all zeros - fix may not be active")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "✓ Observation encoding verified"
else
    echo "ERROR: Observation encoding test failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "All fixes verified and active!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Clean old training data:"
echo "   rm -rf train_dir/blood_v2_warmup_*"
echo ""
echo "2. Start fresh training:"
echo "   python -m blood.train --config=configs/warmup.yaml"
echo ""
echo "3. Monitor DingQue distribution in logs"
echo "   Should see roughly 33/33/33 split instead of 0/0/100"
echo ""