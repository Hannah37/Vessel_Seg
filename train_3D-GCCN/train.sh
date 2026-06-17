#!/bin/bash
#SBATCH --job-name=3dgccn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10-00:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate
module load CUDA/12.1 || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_3D-GCCN

mkdir -p logs runs

export GCCN_ROOT="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_3D-GCCN/3D-GCCN"
export PYTHONPATH="${GCCN_ROOT}:${PYTHONPATH:-}"

python - <<'PY'
from net_model import Encoder3D, Decoder3D, GAT3D
print("OK: 3D-GCCN imports")
PY

JSON="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_3D-GCCN/jsons/sbo_vessel_5fold.json"

FOLD="${SLURM_ARRAY_TASK_ID}"
OUT_DIR="./runs/fold${FOLD}"

echo "=========================================="
echo "3D-GCCN SBO vessel training"
echo "Fold: ${FOLD}"
echo "JSON: ${JSON}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "Start time: $(date)"
echo "=========================================="

python -u train.py \
  --json "${JSON}" \
  --fold "${FOLD}" \
  --out_dir "${OUT_DIR}" \
  --max_epochs 1000 \
  --val_every 5 \
  --full_val_every 100 \
  --roi_x 128 \
  --roi_y 128 \
  --roi_z 128 \
  --batch_size 1 \
  --sw_batch_size 4 \
  --num_workers 4 \
  --lr 1e-3 \
  --weight_decay 5e-4 \
  --base_channels 8 \
  --gnn_hidden 8 \
  --gnn_heads 3 \
  --gnn_dropout 0.2 \
  --graph_stride 12 \
  --edge_dist 16 \
  --gnn_loss_weight 1.0 \
  --amp

echo "=========================================="
echo "Finished fold ${FOLD}"
echo "End time: $(date)"
echo "=========================================="