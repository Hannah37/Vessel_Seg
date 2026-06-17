#!/bin/bash

#SBATCH --job-name=unet_vessel
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
#SBATCH --error=./logs/train101_fold%A_%a.err
#SBATCH --output=./logs/train101_fold%A_%a.out
#SBATCH --time=10-00:00:00

mkdir -p logs

source ~/.bashrc
source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

export nnUNet_raw=/data/drdcad/datasets_nnUNet/nnUNet_raw
export nnUNet_preprocessed=/data/drdcad/datasets_nnUNet/nnUNet_preprocessed
export nnUNet_results=/data/drdcad/datasets_nnUNet/nnUNet_results

DATASET_ID=101
CONFIG=3d_fullres
FOLD=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "nnUNet training"
echo "=========================================="
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
echo "DATASET_ID=$DATASET_ID"
echo "CONFIG=$CONFIG"
echo "FOLD=$FOLD"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=========================================="

nvidia-smi

nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -device cuda

echo "=========================================="
echo "Training finished: fold ${FOLD}"
echo "=========================================="