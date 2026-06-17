#!/bin/bash
#SBATCH --job-name=mednext_sbo
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

# MedNeXt repo를 pip install -e 했다면 없어도 되지만,
# import error 방지를 위해 추가
export PYTHONPATH="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/MedNeXt:${PYTHONPATH:-}"

cd /vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_MedNeXt

mkdir -p logs runs

JSON="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_MedNeXt/jsons/sbo_vessel_5fold.json"
FOLD=${SLURM_ARRAY_TASK_ID}
OUT_DIR="./runs/fold${FOLD}"

echo "=========================================="
echo "MedNeXt SBO vessel training"
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
  --deep_supervision \
  --amp

echo "=========================================="
echo "Finished fold ${FOLD}"
echo "End time: $(date)"
echo "=========================================="