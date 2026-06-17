#!/bin/bash
#SBATCH --job-name=mednext_f3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate
module load CUDA/12.1 || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/MedNeXt:${PYTHONPATH:-}"

cd /vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_MedNeXt

mkdir -p logs runs

JSON="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_MedNeXt/jsons/sbo_vessel_5fold.json"
FOLD=3
OUT_DIR="./runs/fold${FOLD}_stable"

echo "=========================================="
echo "MedNeXt SBO vessel training - stable retry"
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
  --preprocessed_dir /data/drdcad/datasets_nnUNet/nnUNet_preprocessed/Dataset101_SBOvessel/nnUNetPlans_3d_fullres \
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
  --weight_decay 1e-4 \
  --model_id S \
  --kernel_size 3 \
  --deep_supervision 

echo "=========================================="
echo "Finished fold ${FOLD}"
echo "End time: $(date)"
echo "=========================================="