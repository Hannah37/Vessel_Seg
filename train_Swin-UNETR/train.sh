#!/bin/bash

#SBATCH --job-name=48swin_unetr
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
#SBATCH --error=./logs/swin_fold%A_%a.err
#SBATCH --output=./logs/swin_fold%A_%a.out
#SBATCH --time=10-00:00:00

set -e

mkdir -p logs
mkdir -p runs

source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

cd /data/drdcad/Hyuna/projects/vessel_seg/code/train_Swin-UNETR

FOLD=${SLURM_ARRAY_TASK_ID}

JSON_PATH="./jsons/sbo_vessel_fold${FOLD}.json"
OUT_DIR="./runs/fold${FOLD}"

echo "=========================================="
echo "Swin UNETR SBO vessel training"
echo "=========================================="
echo "HOSTNAME=$(hostname)"
echo "FOLD=${FOLD}"
echo "JSON_PATH=${JSON_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

which python
python --version
nvidia-smi

python train.py \
    --json "${JSON_PATH}" \
    --fold "${FOLD}" \
    --out_dir "${OUT_DIR}" \
    --max_epochs 2000 \
    --val_every 5 \
    --full_val_every 100 \
    --roi_x 128 \
    --roi_y 128 \
    --roi_z 128 \
    --batch_size 1 \
    --sw_batch_size 4 \
    --num_workers 4 \
    --lr 4e-4 \
    --weight_decay 3e-5 \
    --feature_size 48 \
    --amp
    

echo "=========================================="
echo "Finished fold ${FOLD}"
echo "=========================================="