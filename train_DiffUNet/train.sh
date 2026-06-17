#!/bin/bash
#SBATCH --job-name=diffunet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

source /data/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate
module load CUDA/12.1 || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Official Diff-UNet BTCV code path
export DIFFUNET_ROOT="/data/drdcad/Hyuna/projects/vessel_seg/code/train_DiffUNet/Diff-UNet"

export PYTHONPATH="${DIFFUNET_ROOT}/BTCV:${DIFFUNET_ROOT}/BraTS2020:${PYTHONPATH:-}"

cd /data/drdcad/Hyuna/projects/vessel_seg/code/train_DiffUNet

mkdir -p logs runs

JSON="/data/drdcad/Hyuna/projects/vessel_seg/code/train_DiffUNet/jsons/sbo_vessel_5fold.json"

FOLD=${SLURM_ARRAY_TASK_ID}
OUT_DIR="./runs/fold${FOLD}"


echo "=========================================="
echo "Diff-UNet SBO vessel training"
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
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --diffusion_steps 1000 \
  --sample_steps 10 \
  --noise_schedule linear \
  --base_channels 32 \
  --amp

echo "=========================================="
echo "Finished fold ${FOLD}"
echo "End time: $(date)"
echo "=========================================="