#!/bin/bash
#SBATCH --job-name=lerf-train
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user="mtiwar26@asu.edu"
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---- EDIT THESE ----
REPO_ROOT=/scratch/$USER/lerf
SCENE=flowers_take7
BACKEND=gaussian
RUN_NAME=flowers_gs_a100
STEPS=30000
SAVE_EVERY=2000
PYTHON_BIN=python
# --------------------

cd "$REPO_ROOT"
mkdir -p logs

echo "Running on host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Scene: $SCENE"
echo "Backend: $BACKEND"
echo "Run: $RUN_NAME"

"$PYTHON_BIN" scripts/train_remote_scene.py \
  --data "data/${SCENE}" \
  --backend "${BACKEND}" \
  --device cuda \
  --steps "${STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --run-name "${RUN_NAME}"
