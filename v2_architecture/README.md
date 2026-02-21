# Mahjong Blood V2 Architecture

This directory contains the completely redesigned AI system for Bloody Battle Mahjong, aimed at **Superhuman** performance. 

## Key Components

1. **`env_core/` (Rust)**
   - High-throughput, batched simulation engine.
   - ISMCE (Inference-time search) routines.
   - Exposes environment bindings to Python via PyO3.

2. **`algo_core/` (Python/PyTorch)**
   - PPO (Actor-Critic) implementation with Generalized Advantage Estimation (GAE).
   - Knowledge distillation from perfect-information Oracle models.
   - Training loop and League System management.

Read the full design doc at [`ARCHITECTURE.md`](./ARCHITECTURE.md).
