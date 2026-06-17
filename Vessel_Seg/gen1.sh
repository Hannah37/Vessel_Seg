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

# =========================
# Change only here
# =========================
SUBJECT_ID="00099"
STUDY_ID="00001"
SERIES_ID="3"

# =========================
# Automatically generated filename/path
# =========================
CASE_ID="${SUBJECT_ID}_${STUDY_ID}_${SERIES_ID}"

CT="/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis/${SUBJECT_ID}/${STUDY_ID}/${CASE_ID}.nii.gz"
# SEG="${CASE_ID}_sma_smv.nii.gz"
SEG="/data/drdcad/Hyuna/projects/vessel_seg/data/gt_sma_smv_nnunet/${CASE_ID}_sma_smv.nii.gz"
OUT="${CASE_ID}_gt_algo_1.nii.gz"
BOWEL_SEG="/data/drdcad/Hyuna/projects/vessel_seg/data/totalseg_output/${CASE_ID}_small_bowel.nii.gz"

python -u gen_vessels1.py \
  --ct "$CT" \
  --seg "$SEG" \
  --out "$OUT" \
  --hu_min 50 \
  --hu_max 370 \
  --vesselness_thr 0.001 \
  --margin 95 \
  --max_iter 700 \
  --min_size 25 \
  --air_hu -500 \
  --air_radius 2 \
  --bowel_seg "$BOWEL_SEG" \
  --bowel_interior_radius 2