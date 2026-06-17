#!/bin/bash

#SBATCH --job-name=unet_smasmv_train
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=./logs/train.err
#SBATCH --output=./logs/train.out
#SBATCH --time=10-00:00:00

mkdir -p logs

source ~/.bashrc
source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

export nnUNet_raw=/data/drdcad/datasets_nnUNet/nnUNet_raw
export nnUNet_preprocessed=/data/drdcad/datasets_nnUNet/nnUNet_preprocessed
export nnUNet_results=/data/drdcad/datasets_nnUNet/nnUNet_results

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

nnUNetv2_train 100 3d_fullres 0 -device cuda