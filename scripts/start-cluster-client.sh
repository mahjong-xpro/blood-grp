#!/bin/bash
set -e

# Usage: ./start-cluster-client.sh <MASTER_IP> <Start_GPU_ID> <End_GPU_ID>
# Example: ./start-cluster-client.sh 192.168.1.100 0 7

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <MASTER_IP> <Start_GPU_ID> <End_GPU_ID>"
    echo "Example: $0 192.168.1.100 0 7"
    exit 1
fi

MASTER_IP=$1
START_GPU=$2
END_GPU=$3

# Ensure running from project root
cd "$(dirname "$0")/.."

# Set config path
export MORTAL_CFG=mortal/config.toml

# Override Server IP in config dynamically using sed (temporary for this run)
# Actually, better to pass as CLI arg if supported, or assume config is synced.
# Since config.toml has [online.remote] host='127.0.0.1', we need to change it.
# We will use a sed command to patch config.toml temporarily if needed, 
# but better to assume the user manually updates config.toml or we provide a way.

# Wait, the best way is to set host in config.toml to 0.0.0.0 on server, 
# and master IP on clients.
# But config.toml is shared? 
# If shared file system: Create config_client.toml?
# If git synced: each node has same file.

# RECOMMENDATION: 
# On Clients, replace '127.0.0.1' with Master IP in config.toml before running.
# OR use a temp config file.

for gpu in $(seq $START_GPU $END_GPU); do
    echo "Launching Client on GPU $gpu..."
    
    # Create unique config for this GPU worker to isolate log directories
    CLIENT_CFG="mortal/config_client_${gpu}.toml"
    cp mortal/config.toml "$CLIENT_CFG"
    
    # 1. Update Master IP
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/host = '127.0.0.1'/host = '$MASTER_IP'/g" "$CLIENT_CFG"
        # 2. Update Log Directory to be unique (e.g. data/mortal/train_play_0)
        # Assuming config has: log_dir = 'data/mortal/train_play'
        sed -i '' "s|data/mortal/train_play|data/mortal/train_play_${gpu}|g" "$CLIENT_CFG"
    else
        sed -i "s/host = '127.0.0.1'/host = '$MASTER_IP'/g" "$CLIENT_CFG"
        sed -i "s|data/mortal/train_play|data/mortal/train_play_${gpu}|g" "$CLIENT_CFG"
    fi
    
    # Run in background with unique config
    MORTAL_CFG="$CLIENT_CFG" CUDA_VISIBLE_DEVICES=$gpu python3 mortal/client.py &
done

wait
