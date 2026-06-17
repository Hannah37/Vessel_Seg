#!/bin/bash
#SBATCH --job-name=watch_smasmv_vessel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
shopt -s nullglob

source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

CODE_DIR="/vf/users/drdcad/Hyuna/projects/vessel_seg/code/Vessel_Seg"
cd "$CODE_DIR"

mkdir -p logs

CT_ROOT="/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis"

SEG_DIR_GT="/data/drdcad/Hyuna/projects/vessel_seg/data/gt_sma_smv_nnunet"
SEG_DIR_PRED="/data/drdcad/Hyuna/projects/vessel_seg/data/pred_sma_smv"

SEG_DIRS=(
    "$SEG_DIR_GT"
    "$SEG_DIR_PRED"
)

BOWEL_DIR="/data/drdcad/Hyuna/projects/vessel_seg/data/totalseg_output"
OUT_DIR="/data/drdcad/Hyuna/projects/vessel_seg/data/gt_vessel_algo"

mkdir -p "$OUT_DIR"

LOCK_DIR="$OUT_DIR/.locks_watch"
DONE_DIR="$OUT_DIR/.done_watch"
FAILED_DIR="$OUT_DIR/.failed_watch"

mkdir -p "$LOCK_DIR" "$DONE_DIR" "$FAILED_DIR"

FAILED_LIST="$OUT_DIR/failed_cases_watch.txt"
MISSING_LIST="$OUT_DIR/missing_input_cases_watch.txt"

touch "$FAILED_LIST"
touch "$MISSING_LIST"

SLEEP_SEC=60

echo "SEG_DIR_GT:   $SEG_DIR_GT"
echo "SEG_DIR_PRED: $SEG_DIR_PRED"
echo "BOWEL_DIR:    $BOWEL_DIR"
echo "OUT_DIR:      $OUT_DIR"
echo "Start time: $(date)"
echo "========================================"


file_ready() {
    local f="$1"

    [ -s "$f" ] || return 1

    local s1 s2 s3
    s1=$(stat -c%s "$f" 2>/dev/null || echo 0)
    sleep 20
    s2=$(stat -c%s "$f" 2>/dev/null || echo 0)
    sleep 20
    s3=$(stat -c%s "$f" 2>/dev/null || echo 0)

    if [ "$s1" != "$s2" ] || [ "$s2" != "$s3" ]; then
        return 1
    fi

    gzip -t "$f" >/dev/null 2>&1 || return 1

    return 0
}


get_case_id_from_seg() {
    local filename
    filename=$(basename "$1")

    if [[ "$filename" == *_sma_smv.nii.gz ]]; then
        echo "${filename%_sma_smv.nii.gz}"
    else
        echo "${filename%.nii.gz}"
    fi
}


process_one_case() {
    local seg="$1"
    local source_tag="$2"

    local case_id
    case_id=$(get_case_id_from_seg "$seg")

    local out="$OUT_DIR/${case_id}_gt.nii.gz"

    # source별 done/failed를 따로 둠.
    # gt 실패해도 pred가 나중에 같은 case를 다시 처리할 수 있게 하기 위함.
    local done_file="$DONE_DIR/${source_tag}_${case_id}.done"
    local failed_file="$FAILED_DIR/${source_tag}_${case_id}.failed"
    local lock_dir="$LOCK_DIR/${source_tag}_${case_id}.lock"

    if [ -f "$out" ]; then
        echo "[SKIP] Output already exists: $out"
        touch "$done_file"
        return 0
    fi

    if [ -f "$done_file" ]; then
        echo "[SKIP] Already done: $source_tag $case_id"
        return 0
    fi

    if [ -f "$failed_file" ]; then
        echo "[SKIP] Previously failed: $source_tag $case_id"
        return 0
    fi

    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "[SKIP] Locked by another process: $source_tag $case_id"
        return 0
    fi

    IFS="_" read -r SUBJECT_ID STUDY_ID SERIES_ID <<< "$case_id"

    local ct="${CT_ROOT}/${SUBJECT_ID}/${STUDY_ID}/${case_id}.nii.gz"
    local bowel_seg="${BOWEL_DIR}/${case_id}_small_bowel.nii.gz"

    echo "========================================"
    echo "SOURCE:    $source_tag"
    echo "CASE_ID:   $case_id"
    echo "CT:        $ct"
    echo "SEG:       $seg"
    echo "BOWEL_SEG: $bowel_seg"
    echo "OUT:       $out"

    if [ ! -f "$ct" ]; then
        echo "[SKIP] Missing CT: $ct"
        echo "$source_tag $case_id missing CT" >> "$MISSING_LIST"
        touch "$failed_file"
        rmdir "$lock_dir"
        return 0
    fi

    if [ ! -f "$seg" ]; then
        echo "[SKIP] Missing SMA/SMV seg: $seg"
        echo "$source_tag $case_id missing SEG" >> "$MISSING_LIST"
        touch "$failed_file"
        rmdir "$lock_dir"
        return 0
    fi

    if [ ! -f "$bowel_seg" ]; then
        echo "[SKIP] Missing small bowel seg: $bowel_seg"
        echo "$source_tag $case_id missing BOWEL_SEG" >> "$MISSING_LIST"
        touch "$failed_file"
        rmdir "$lock_dir"
        return 0
    fi

    if python -u gen_vessels1.py \
      --ct "$ct" \
      --seg "$seg" \
      --out "$out" \
      --hu_min 50 \
      --hu_max 370 \
      --vesselness_thr 0.001 \
      --margin 95 \
      --max_iter 700 \
      --min_size 25 \
      --air_hu -500 \
      --air_radius 2 \
      --bowel_seg "$bowel_seg" \
      --bowel_interior_radius 2; then

        echo "[DONE] $source_tag $case_id"
        touch "$done_file"
    else
        echo "[FAILED] $source_tag $case_id"
        echo "$source_tag $case_id" >> "$FAILED_LIST"
        touch "$failed_file"
    fi

    rmdir "$lock_dir"
}


process_dir() {
    local seg_dir="$1"
    local source_tag="$2"
    local require_ready="$3"

    if [ ! -d "$seg_dir" ]; then
        echo "[WARNING] SEG_DIR does not exist: $seg_dir"
        return 0
    fi

    echo "Searching SEG_DIR: $seg_dir"

    for seg in "$seg_dir"/*.nii.gz; do
        [ -e "$seg" ] || continue

        local case_id
        case_id=$(get_case_id_from_seg "$seg")

        local out="$OUT_DIR/${case_id}_gt.nii.gz"

        if [ -f "$out" ]; then
            continue
        fi

        if [ "$require_ready" = "yes" ]; then
            if ! file_ready "$seg"; then
                echo "[WAIT] File not ready yet: $seg"
                continue
            fi
        fi

        process_one_case "$seg" "$source_tag"
    done
}


has_pending_cases() {
    local seg_dir source_tag require_ready seg case_id out done_file failed_file

    for seg_dir in "${SEG_DIRS[@]}"; do
        if [ "$seg_dir" = "$SEG_DIR_GT" ]; then
            source_tag="gt"
        else
            source_tag="pred"
        fi

        [ -d "$seg_dir" ] || continue

        for seg in "$seg_dir"/*.nii.gz; do
            [ -e "$seg" ] || continue

            case_id=$(get_case_id_from_seg "$seg")
            out="$OUT_DIR/${case_id}_gt.nii.gz"
            done_file="$DONE_DIR/${source_tag}_${case_id}.done"
            failed_file="$FAILED_DIR/${source_tag}_${case_id}.failed"

            if [ ! -f "$out" ] && [ ! -f "$done_file" ] && [ ! -f "$failed_file" ]; then
                return 0
            fi
        done
    done

    return 1
}


while true; do
    echo "Watcher pass at: $(date)"

    # 기존 run_all_data.sh와 동일하게 GT 먼저.
    # GT는 이미 완성된 파일이므로 readiness check 불필요.
    process_dir "$SEG_DIR_GT" "gt" "no"

    # pred는 nnUNetv2_predict가 쓰는 중일 수 있으므로 readiness check 필요.
    process_dir "$SEG_DIR_PRED" "pred" "yes"

    if [ -f "$SEG_DIR_PRED/.predict_done" ]; then
        if ! has_pending_cases; then
            echo "Prediction done marker found and no pending cases."
            break
        fi
    fi

    sleep "$SLEEP_SEC"
done

echo "========================================"
echo "Watcher finished at: $(date)"

echo "Output file count:"
find "$OUT_DIR" -maxdepth 1 -name "*_gt.nii.gz" | wc -l

echo "Missing input cases:"
cat "$MISSING_LIST"

echo "Failed cases:"
cat "$FAILED_LIST"