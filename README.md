# CARLA DeepTraffic

Deep reinforcement learning experiments for autonomous highway driving in the [CARLA simulator](https://carla.org/), inspired by DeepTraffic and Sentdex-style DQN training loops.

## Overview

This repository contains:

- A CARLA driving environment wrapper (`CarEnvironment`)
- A DQN agent implementation (`DQNAgent`) with multiple model options
- Training and evaluation scripts for highway lane-following / lane-changing behavior
- Utility scripts for spawn-point discovery, visualization, TensorBoard, and model export
- Saved model artifacts and training logs from multiple experiment runs

Primary workflow:

1. Launch CARLA.
2. Train with `CarlaClient_5.py`.
3. Evaluate trained models with `ViewModel_5.py`.
4. Inspect metrics in TensorBoard.

---

## Repository Structure

Key files:

- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/CarlaClient_5.py`  
  Main training script (current/most feature-rich version).
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/ViewModel_5.py`  
  Runs inference with a trained model inside CARLA.
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/CarlaClient_3.py`  
  Earlier training version.
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/ViewModel_3.py`  
  Earlier model playback script.
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/find_spawns.py`  
  Utility to inspect and iterate CARLA spawn points.
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/spawn_spectate.py`  
  Utility for waypoint/spawn debugging and spectator positioning.
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/export_model.py`  
  Exports architecture diagrams for available networks.
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/run_tensorboard.py`  
  Starts TensorBoard (Windows-oriented helper script).
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/Fundimentals.py`  
  Early CARLA sandbox/demo script.

Experiment outputs:

- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/models/` and `models_*` folders: saved models, Q-value logs, constants snapshots
- `/home/runner/work/CARLA_DeepTraffic/CARLA_DeepTraffic/logs/`: TensorBoard event logs

---

## Environment and Compatibility

The core training code is written against an older ML stack:

- Python `3.6.8`
- TensorFlow `1.14.0`
- Keras `2.2.5` (or `2.2.4` in older scripts)
- h5py `2.10.0`
- NumPy `1.16.4`
- OpenCV `4.5.3`
- CARLA `0.9.13`

> Note: Newer Python/TensorFlow/Keras versions are likely to break this code without migration work.
---

## Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/archiehull/CARLA_DeepTraffic.git
   cd CARLA_DeepTraffic
   ```

2. **Create and activate a compatible Python environment** (recommended: Python 3.6.8).

3. **Install Python dependencies**
   ```bash
   pip install carla tensorflow==1.14.0 keras==2.2.5 h5py==2.10.0 numpy==1.16.4 opencv-python==4.5.3.56 tqdm
   ```

4. **Start CARLA simulator** (default host/port expected by scripts: `localhost:2000`).

---

## Training

Run:

```bash
python CarlaClient_5.py
```

### Important training settings

Inside `CarlaClient_5.py`, you can configure:

- Model type (`MODEL_NAME`): `64x3`, `Xception`, `CNN1`, or `DeepTraffic`
- Episode count and length (`EPISODES`, `EPISODE_LENGTH`)
- Exploration parameters (`epsilon`, `EPSILON_DECAY`, `MIN_EPSILON`)
- Replay/training settings (memory sizes, batch sizes, target update cadence)
- Simulation behavior (traffic population, waypoint generation, autopilot usage)

Training outputs include:

- Saved `.model` checkpoints in `models/`
- TensorBoard logs in `logs/`
- Optional constants and Q-value log snapshots

---

## Model Evaluation / Playback

Run:

```bash
python ViewModel_5.py --model_path "<path_to_trained_model>"
```

If `--model_path` is omitted, the script uses its built-in default path.  
The script can optionally show camera input in an OpenCV window and prints per-step action/Q diagnostics.

---

## TensorBoard

Manual start:

```bash
tensorboard --logdir=logs
```

Then open: `http://localhost:6006`

`run_tensorboard.py` provides a convenience launcher for Windows setups.

---

## Utility Scripts

- `find_spawns.py`  
  Iterate through map spawn points and place a vehicle/spectator for quick location inspection.

- `spawn_spectate.py`  
  Generate lane waypoints, populate highway traffic actors, and visualize selected waypoint sets.

- `export_model.py`  
  Saves architecture diagrams for available model definitions as PNG images.

---

## Known Limitations

- Hardcoded map and spawn assumptions (e.g., Town04 and specific spawn index usage).
- Older TensorFlow/Keras APIs (`tf.compat.v1`, legacy backend/session usage).
- **Legacy API prevent GPU leveraging on Windows 11**
- Several scripts include Windows-style file paths and behavior.
- No formal automated test suite or packaging metadata currently included.

---

## Troubleshooting

- **`ImportError: carla`**  
  Install CARLA Python API in the active environment and ensure version compatibility.

- **TensorFlow/Keras session/backend errors**  
  Use Python 3.6.8 + TensorFlow 1.14 + Keras 2.2.x as expected by the scripts.

- **No connection to simulator**  
  Confirm CARLA is running on `localhost:2000` before starting training/evaluation.

- **Model path load failures in view scripts**  
  Pass a valid path using `--model_path`.

---

## References

- Harrison Kinsley (Sentdex), reinforcement learning with CARLA:
  https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/
- CARLA simulator:
  https://carla.org/

---

# Report + Key Findings

This project investigated Deep Q-Network (DQN) training for autonomous highway navigation in CARLA using image-based observations and a discrete action space consisting of:

- Lane Left
- Lane Right
- Accelerate
- Brake
- Maintain Speed

## What Worked

- Built a complete CARLA training pipeline including experience replay, target network updates, epsilon-greedy exploration, modular reward shaping, and model checkpointing.
- Successfully configured a multi-lane highway scenario in Town04 with dynamic traffic.
- Implemented multiple CNN backbones and made architecture swapping straightforward for experimentation.

## Main Technical Outcome

The **64x3 CNN** architecture proved to be the most stable under CPU-only constraints.

Compared with the Xception and CNN1 architectures, it demonstrated:

- Better stability in reward trends
- More bounded Q-values
- Lower volatility during training

## Core Limitation

Real-time training and inference were not achieved due to TensorFlow/CUDA/Windows compatibility issues, which forced CPU-only execution.

This significantly reduced action throughput and destabilised learning, particularly for the larger network architectures.

---

# Limitations and Lessons Learned

## Practical Constraints Encountered

- GPU acceleration was unavailable with the selected software stack on the target machine.
- CPU-bound training introduced high prediction/training latency and a reduced control frequency.
- Multithreaded training sometimes increased apparent action throughput but could also lead to stale or unsynchronised Q-value updates.

## Observed Training Behaviour

- Larger and deeper models, particularly the Xception-based network, were poorly matched to the available hardware.
- Some runs exhibited extreme loss values and Q-value spikes, likely associated with resource contention during model checkpoint export.
- Simpler CNN architectures consistently produced more stable learning under constrained compute resources.

## Key Takeaway

For CARLA + DQN experiments on limited hardware, model simplicity and stable system timing can be more important than architectural complexity.

---

# Reproducibility Notes

To improve reproducibility and debugging, this project includes deterministic-style reset behaviour where possible, episode-level logging, periodic Q-value logging, debug flags, and model export checkpoints.

When reproducing results, the following should remain fixed:

- Map and spawn configuration
- Reward function constants
- Network architecture used for each experiment
- Hardware and software environment (OS, CPU/GPU, TensorFlow/CUDA versions)

---

# Future Work

Potential directions for extending this work include:

- Migrating to a modern reinforcement learning stack with GPU-compatible tooling or a Linux-based training environment.
- Comparing DQN against Double DQN, Dueling DQN, and policy-gradient methods.
- Adding richer observations through stacked image frames and vehicle telemetry fusion.
- Introducing curriculum-based reward shaping, progressing from lane keeping to safe overtaking and speed optimisation.
- Evaluating robustness across varying weather, lighting conditions, and traffic densities.
- Implementing automated hyperparameter sweeps and experiment tracking.

---

# Contribution of This Work

Although sustained real-time autonomous behaviour was not achieved, this project contributes:

- A modular CARLA + DQN experimentation framework
- Documented evidence of compute-bound reinforcement learning failure modes
- A practical **64x3 CNN** baseline for constrained hardware environments

This repository is intended to serve both as:

- A foundation for future autonomous-driving reinforcement learning experiments
- A case study demonstrating the trade-offs between model complexity and system-level feasibility

---

# At a Glance

| Item | Summary |
|------|---------|
| **Best-performing model** | 64x3 CNN |
| **Environment** | CARLA Town04 highway scenario |
| **Action space** | 5 discrete actions |
| **Main blocker** | GPU/toolchain incompatibility resulting in CPU-only training |
| **Outcome** | Functional training pipeline with partial learning success, but no sustained real-time autonomous driving |
