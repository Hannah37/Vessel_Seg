#!/bin/bash
#SBATCH --job-name=branch_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run gen_vessels1.py for all SMA/SMV files from:
# 1) gt_sma_smv_nnunet
# 2) pred_sma_smv
# to generate full vessel ground truths.

set -uo pipefail
shopt -s nullglob

source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

CODE_DIR="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/Vessel_Seg"
cd "$CODE_DIR"

mkdir -p logs

CT_ROOT="/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis"

SEG_DIR_GT="/data/drdcad/Hyuna/projects/vessel_seg/data/gt_sma_smv_nnunet"
SEG_DIR_PRED="/data/drdcad/Hyuna/projects/vessel_seg/data/pred_sma_smv"

# gt_sma_smv_nnunet을 먼저 둠.
# 같은 CASE_ID가 두 폴더에 있으면 gt_sma_smv_nnunet 버전을 우선 사용.
SEG_DIRS=(
    "$SEG_DIR_GT"
    "$SEG_DIR_PRED"
)

BOWEL_DIR="/data/drdcad/Hyuna/projects/vessel_seg/data/totalseg_output"
OUT_DIR="/data/drdcad/Hyuna/projects/vessel_seg/data/gt_vessel_algo"

mkdir -p "$OUT_DIR"

FAILED_LIST="$OUT_DIR/failed_cases.txt"
MISSING_LIST="$OUT_DIR/missing_input_cases.txt"
DUPLICATE_LIST="$OUT_DIR/duplicate_seg_cases.txt"

rm -f "$FAILED_LIST" "$MISSING_LIST" "$DUPLICATE_LIST"

echo "SEG_DIR_GT:   $SEG_DIR_GT"
echo "SEG_DIR_PRED: $SEG_DIR_PRED"
echo "BOWEL_DIR:    $BOWEL_DIR"
echo "OUT_DIR:      $OUT_DIR"
echo "Start time: $(date)"
echo "========================================"

declare -A SEEN_CASES

for SEG_DIR in "${SEG_DIRS[@]}"; do
    echo "Searching SEG_DIR: $SEG_DIR"

    if [ ! -d "$SEG_DIR" ]; then
        echo "[WARNING] SEG_DIR does not exist: $SEG_DIR"
        continue
    fi

    for SEG in "$SEG_DIR"/*.nii.gz; do
        filename=$(basename "$SEG")

        if [[ "$filename" == *_sma_smv.nii.gz ]]; then
            CASE_ID="${filename%_sma_smv.nii.gz}"
        else
            CASE_ID="${filename%.nii.gz}"
        fi

        SEEN_CASES[$CASE_ID]="$SEG"

        IFS="_" read -r SUBJECT_ID STUDY_ID SERIES_ID <<< "$CASE_ID"

        CT="${CT_ROOT}/${SUBJECT_ID}/${STUDY_ID}/${CASE_ID}.nii.gz"
        BOWEL_SEG="${BOWEL_DIR}/${CASE_ID}_small_bowel.nii.gz"
        OUT="${OUT_DIR}/${CASE_ID}_gt.nii.gz"

        echo "========================================"
        echo "CASE_ID:   $CASE_ID"
        echo "SEG_DIR:   $SEG_DIR"
        echo "CT:        $CT"
        echo "SEG:       $SEG"
        echo "BOWEL_SEG: $BOWEL_SEG"
        echo "OUT:       $OUT"

        if [ ! -f "$CT" ]; then
            echo "[SKIP] Missing CT: $CT"
            echo "$CASE_ID missing CT" >> "$MISSING_LIST"
            continue
        fi

        if [ ! -f "$SEG" ]; then
            echo "[SKIP] Missing SMA/SMV seg: $SEG"
            echo "$CASE_ID missing SEG" >> "$MISSING_LIST"
            continue
        fi

        if [ ! -f "$BOWEL_SEG" ]; then
            echo "[SKIP] Missing small bowel seg: $BOWEL_SEG"
            echo "$CASE_ID missing BOWEL_SEG" >> "$MISSING_LIST"
            continue
        fi

        if [ -f "$OUT" ]; then
            echo "[SKIP] Output already exists: $OUT"
            continue
        fi

        if ! python -u gen_vessels1.py \
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
          --bowel_interior_radius 2; then

            echo "[FAILED] $CASE_ID"
            echo "$CASE_ID" >> "$FAILED_LIST"
            continue
        fi

        echo "[DONE] $CASE_ID"
    done
done

echo "========================================"
echo "Finished time: $(date)"

echo "Output file count:"
find "$OUT_DIR" -maxdepth 1 -name "*_gt.nii.gz" | wc -l

echo "Missing input cases:"
if [ -f "$MISSING_LIST" ]; then
    cat "$MISSING_LIST"
else
    echo "None"
fi

echo "Duplicate SEG cases:"
if [ -f "$DUPLICATE_LIST" ]; then
    cat "$DUPLICATE_LIST"
else
    echo "None"
fi

echo "Failed cases:"
if [ -f "$FAILED_LIST" ]; then
    cat "$FAILED_LIST"
else
    echo "None"
fi