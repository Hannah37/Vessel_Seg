#!/bin/bash
#SBATCH --job-name=effi_mednext
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

cd /vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_EfficientMedNeXt
mkdir -p logs runs jsons

# Official EfficientMedNeXt repo location.
# Clone once before sbatch if this folder does not exist:
# git clone https://github.com/SLDGroup/EfficientMedNeXt.git /vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_EfficientMedNeXt/EfficientMedNeXt
export EFFICIENT_MEDNEXT_ROOT="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_EfficientMedNeXt/EfficientMedNeXt"
export PYTHONPATH="${EFFICIENT_MEDNEXT_ROOT}:${PYTHONPATH:-}"

python - <<'PY'
from networks.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext
model = create_efficient_mednext(
    1,
    1,
    'S',
    n_channels=32,
    kernel_sizes=[1, 3, 5],
    strides=[1, 1, 1],
    uniform_dec_channels=32,
    deep_supervision=False,
)
print("OK: EfficientMedNeXt imports")
print("Params:", sum(p.numel() for p in model.parameters()))
PY

JSON="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_EfficientMedNeXt/jsons/sbo_vessel_5fold.json"
FOLD="${SLURM_ARRAY_TASK_ID}"
OUT_DIR="./runs/fold${FOLD}"

echo "=========================================="
echo "EfficientMedNeXt SBO vessel training"
echo "Fold: ${FOLD}"
echo "JSON: ${JSON}"
echo "OUT_DIR: ${OUT_DIR}"
echo "EFFICIENT_MEDNEXT_ROOT: ${EFFICIENT_MEDNEXT_ROOT}"
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
  --num_patch_samples 1 \
  --sw_batch_size 4 \
  --num_workers 4 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --model_size S \
  --feature_size 32 \
  --n_decoder_channels 32 \
  --kernel_sizes 1 3 5 \
  --amp

echo "=========================================="
echo "Finished fold ${FOLD}"
echo "End time: $(date)"
echo "=========================================="
