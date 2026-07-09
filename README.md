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
