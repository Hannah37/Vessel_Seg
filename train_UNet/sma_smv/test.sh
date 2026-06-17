#!/bin/bash

#SBATCH --job-name=unet_smasmv_test
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=./logs/predict.err
#SBATCH --output=./logs/predict.out
#SBATCH --time=10-00:00:00

source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

export nnUNet_raw="/data/drdcad/datasets_nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/data/drdcad/datasets_nnUNet/nnUNet_preprocessed"
export nnUNet_results="/data/drdcad/datasets_nnUNet/nnUNet_results"

nnUNetv2_predict \
  -i /data/drdcad/datasets_nnUNet/nnUNet_raw/Dataset100_SmallBowelObstruction/imagesTs \
  -o /data/drdcad/Hyuna/projects/vessel_seg/data/pred_sma_smv \
  -d 100 \
  -c 3d_fullres \
  -f 0 \
  -chk checkpoint_best.pth \
  -device cuda

