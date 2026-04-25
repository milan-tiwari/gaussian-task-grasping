# SOL Training Workflow

This project is easiest to run on SOL as a pure scene-training job first, then bring the trained artifacts back to the robot workspace.

## What To Upload

Upload these items to your SOL working directory:

- this repo checkout
- the scene folders you want under `data/`

Recommended layout on SOL:

```text
lerf/
  data/
    flowers_take7/
    cords_take1/
    ...
  scripts/
    train_remote_scene.py
  robot_lerf/
  pyproject.toml
  README.md
```

## Environment

Use a Nerfstudio environment on SOL with CUDA available. The remote script only needs Nerfstudio and its dependencies to train scene models.

If you also want to import local modules from this repo later, install the repo in editable mode:

```bash
python -m pip install -e .
```

## Training Commands

Gaussian Splatting on a GPU:

```bash
python scripts/train_remote_scene.py \
  --data data/flowers_take7 \
  --backend gaussian \
  --device cuda \
  --steps 30000 \
  --run-name flowers_gs_a100
```

Original LERF baseline on a GPU:

```bash
python scripts/train_remote_scene.py \
  --data data/flowers_take7 \
  --backend nerfstudio \
  --device cuda \
  --steps 3000 \
  --run-name flowers_lerf_baseline
```

For a very short smoke test on SOL:

```bash
python scripts/train_remote_scene.py \
  --data data/flowers_take7 \
  --backend gaussian \
  --device cuda \
  --steps 10 \
  --eval-every 0 \
  --run-name flowers_gs_smoke
```

## Resume Behavior

The remote training script automatically resumes from the latest checkpoint in the same run directory if it exists.

That means if you rerun:

```bash
python scripts/train_remote_scene.py \
  --data data/flowers_take7 \
  --backend gaussian \
  --device cuda \
  --steps 30000 \
  --run-name flowers_gs_a100
```

and `outputs/flowers_take7/splatfacto/flowers_gs_a100/nerfstudio_models/step-*.ckpt` already exists, the job will continue from the newest checkpoint instead of starting over.

Useful options:

- `--save-every 500` for more frequent checkpoints
- `--keep-all-checkpoints` if you want every checkpoint kept instead of only the latest one
- `--no-resume` if you intentionally want to restart the same run name from scratch

If you prefer submitting through Slurm, use the template at `scripts/train_scene.sbatch`:

```bash
sbatch scripts/train_scene.sbatch
```

The template is written like a normal sbatch file: edit the variables near the top of the file and set `REPO_ROOT` to your SOL checkout path, for example `/scratch/$USER/lerf`.

Because Slurm opens the log files before the script body runs, create the log directory once before the first submit:

```bash
cd /scratch/$USER/lerf
mkdir -p logs
sbatch scripts/train_scene.sbatch
```

Before submitting, edit:

- the cluster-specific header lines if your GPU allocation uses a different partition, qos, or mail setup
- `REPO_ROOT`
- `SCENE`
- `RUN_NAME`
- `PYTHON_BIN`

## What To Download Back

Download the whole run directory for each trained scene:

```text
outputs/<scene_name>/<method_name>/<run_name>/
```

The important files inside are:

- `config.yml`
- `nerfstudio_models/step-*.ckpt`
- training logs written in the run directory

Example:

```text
outputs/flowers_take7/splatfacto/flowers_gs_a100/
```

## Important Portability Note

Use repo-relative dataset paths like `data/flowers_take7` when launching training.

The training script saves that relative path into `config.yml`, which means the model stays portable as long as your local workspace also has the same dataset under `data/<scene_name>/`.

If you train with an absolute path on SOL, you will usually need to edit the saved config before loading it locally.

## Suggested Transfer Commands

Upload:

```bash
rsync -av --progress /local/path/to/lerf/ your_username@sol.asu.edu:/path/to/lerf/
```

Download one finished run:

```bash
rsync -av --progress \
  your_username@sol.asu.edu:/path/to/lerf/outputs/flowers_take7/splatfacto/flowers_gs_a100/ \
  /local/path/to/lerf/outputs/flowers_take7/splatfacto/flowers_gs_a100/
```

## After Downloading

Keep these on your local machine:

- the original scene folder under `data/<scene_name>/`
- the downloaded trained run under `outputs/<scene_name>/<method_name>/<run_name>/`

That is the minimum layout needed for later loading through Nerfstudio config paths.

## Reusing SfM Points In Robot Frame

If an HLOC/COLMAP-backed scene reconstructs better than the original robot-pose scene, you can transfer its sparse
point cloud back into the original robot/world frame and retrain there.

Use:

```bash
python scripts/prepare_robot_frame_scene.py \
  --robot-scene data/flowers_take7 \
  --sfm-scene data/flowers_take7_hloc_v3 \
  --output-scene data/flowers_take7_robotinit \
  --overwrite
```

This creates a new scene folder that:

- keeps the original robot/world-frame `transforms.json`
- adds an aligned `sparse_pc.ply`
- writes `alignment.json` with the fitted similarity transform and camera error statistics

Then retrain on the aligned robot-frame scene:

```bash
python scripts/train_remote_scene.py \
  --data data/flowers_take7_robotinit \
  --backend gaussian \
  --device cuda \
  --steps 30000 \
  --save-every 2000 \
  --run-name flowers_gs_robotinit
```

This is the clean bridge between "better SfM reconstruction" and "scene model still lives in the robot frame."
