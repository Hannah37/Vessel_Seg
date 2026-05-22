#!/bin/bash
#SBATCH --job-name=sma_branch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

cd /vf/users/drdcad/Hyuna/projects/vessel_seg/code/Vessel_Seg

python -u gen_vessels_2.py  \
  --hu_min 60 \
  --vesselness_thr 0.0005 \
  --hu_max 380 \
  --max_radius_vox 6 \
  --margin 120 \
  --max_iter 1000 \
  --min_size 2