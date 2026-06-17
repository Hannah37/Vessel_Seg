#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=./logs/preprocess.err
#SBATCH --output=./logs/preprocess.out
#SBATCH --time=10-00:00:00

source ~/.bashrc
source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

export nnUNet_raw=/data/drdcad/datasets_nnUNet/nnUNet_raw
export nnUNet_preprocessed=/data/drdcad/datasets_nnUNet/nnUNet_preprocessed
export nnUNet_results=/data/drdcad/datasets_nnUNet/nnUNet_results

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

ls $nnUNet_raw
ls $nnUNet_raw/Dataset101_SBOvessel


nnUNetv2_plan_and_preprocess \
-d 101 \
--verify_dataset_integrity \
--clean \
-c 3d_fullres