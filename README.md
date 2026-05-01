# Gaussian Task Grasping: Gaussian Splatting for Task-Oriented Grasping

> This repo builds on the LERF-TOGO research direction with a Gaussian Splatting scene backend, robot-frame alignment tooling, and a working grasp-generation UI for semantic robotic manipulation.

<p align="center">
  <img src="docs/media/flowers_take7_hloc_v3_demo.gif" alt="HLOC-initialized Gaussian reconstruction demo" width="820" />
</p>

## What This Repo Shows

- Gaussian Splatting scenes train and load through a shared scene backend abstraction in `robot_lerf/scene_backends.py`.
- HLOC/COLMAP reconstructions can be aligned back into the robot frame with `scripts/prepare_robot_frame_scene.py`.
- `scripts/gen_grasp.py` loads a trained scene, builds the world-frame point cloud, generates grasps, runs collision filtering, and serves a `viser` UI.
- The active project path is Gaussian geometry plus grasp generation, with semantic integration still treated as experimental work rather than the default runtime path.

## Engineering Contributions In This Snapshot

- Backend abstraction to support both the original LERF/Nerfstudio path and a Gaussian Splatting path from the same UI.
- Robot-frame alignment workflow for transferring high-quality SfM geometry back into executable robot coordinates.
- Scene loading, workspace alignment, and semantic-query debugging utilities around the Gaussian grasping path.
- Remote training and embedding helpers for larger scene experiments.
- Compatibility fixes across the viewer and vendored GraspNet/Dex-Net stack so the app boots and runs on current environments.

## Quick Launch

### 1. Install Nerfstudio first

Follow the official Nerfstudio setup until `tinycudann` is working:

- [Nerfstudio installation guide](https://docs.nerf.studio/en/latest/quickstart/installation.html)

### 2. Clone and install this repo

```bash
git clone https://github.com/milan-tiwari/gaussian-task-grasping.git
cd gaussian-task-grasping
python -m pip install -e .
```

### 3. Install the native GraspNet modules

```bash
cd robot_lerf/graspnet_baseline/knn
python setup.py install

cd ../pointnet2
python setup.py install

cd ../graspnetAPI
pip install -e .

cd ../../..
```

### 4. Launch the interactive grasp UI

```bash
python scripts/gen_grasp.py \
  --config-path outputs/<scene_name>/<method>/<run_name>/config.yml \
  --scene-backend gaussian \
  --viser-port 8080
```

Then open `http://127.0.0.1:8080`.

Query format:

- `grasp` / `twist`: `object;part`
- `pick & place` / `pour`: `object;part;place`
- Example: `flower;petal`

## Important Runtime Notes

- Gaussian training is effectively CUDA-only in this Nerfstudio/Splatfacto setup.
- CPU-only Macs are fine for documentation, dataset prep, and baseline repo work, but not for realistic Gaussian training.
- The current Gaussian backend can load the checkpoint, build the point cloud, generate grasps, and serve the UI.
- Semantic-query work is still experimental and is not the main advertised launch path for this repo snapshot.

## Key Files

| Goal | File |
| --- | --- |
| Launch or retrain a scene run | `scripts/train_remote_scene.py` |
| Align an HLOC/SfM scene back into robot frame | `scripts/prepare_robot_frame_scene.py` |
| Run the main grasp UI | `scripts/gen_grasp.py` |
| Build Gaussian point clouds for grasping | `robot_lerf/graspnet_baseline/load_ns_model.py` |
| Shared scene backend abstraction | `robot_lerf/scene_backends.py` |

## Expected Repo Layout

```text
gaussian-task-grasping/
  data/
    <scene_name>/
      transforms.json
      images...
      depth...
  outputs/
    <scene_name>/
      <method>/<run_name>/config.yml
  robot_lerf/
  scripts/
```

## Citation

If this repo helps your work, please cite the original paper:

```bibtex
@inproceedings{lerftogo2023,
  title={Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping},
  author={Adam Rashid and Satvik Sharma and Chung Min Kim and Justin Kerr and Lawrence Yunliang Chen and Angjoo Kanazawa and Ken Goldberg},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023},
}
```
