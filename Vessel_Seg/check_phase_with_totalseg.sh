#!/bin/bash
#SBATCH --job-name=check_phase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

mkdir -p logs

source /vf/users/drdcad/Hyuna/projects/vessel_seg/seg/bin/activate

# =========================
# Paths
# =========================
EXCEL="/data/drdcad/Hyuna/projects/vessel_seg/data/5samples_updated.xlsx"
SHEET="used_data"

CT_ROOT="/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis"
OUT_DIR="/data/drdcad/Hyuna/projects/vessel_seg/data/contrast_phase_check"

mkdir -p "$OUT_DIR"

CASE_LIST="${OUT_DIR}/used_data_case_list.txt"
MISSING_LIST="${OUT_DIR}/missing_ct_cases.txt"
FAILED_LIST="${OUT_DIR}/failed_phase_cases.txt"

SUMMARY_CSV="${OUT_DIR}/phase_summary.csv"
SUMMARY_XLSX="${OUT_DIR}/phase_summary.xlsx"

rm -f "$CASE_LIST" "$MISSING_LIST" "$FAILED_LIST" "$SUMMARY_CSV" "$SUMMARY_XLSX"

echo "EXCEL:   $EXCEL"
echo "SHEET:   $SHEET"
echo "CT_ROOT: $CT_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "Start time: $(date)"
echo "========================================"

# =========================
# 1) Extract CASE_IDs from used_data sheet
# =========================
python - <<EOF
import pandas as pd
import re

excel = "$EXCEL"
sheet = "$SHEET"
out = "$CASE_LIST"

df = pd.read_excel(excel, sheet_name=sheet)

print("Columns:", list(df.columns))

cols = {str(c).strip().lower(): c for c in df.columns}

def find_col(candidates):
    for cand in candidates:
        cand = cand.lower()
        for k, orig in cols.items():
            if k == cand or cand in k:
                return orig
    return None

case_col = find_col(["case_id", "caseid", "case"])
subject_col = find_col(["subject_id", "subject", "patient_id"])
study_col = find_col(["study_id", "study"])
series_col = find_col(["series_id", "series"])

case_ids = []

if case_col is not None:
    print(f"Using CASE_ID column: {case_col}")
    for v in df[case_col].dropna():
        s = str(v).strip()
        m = re.search(r"(\\d{5})_(\\d{5})_(\\d+)", s)
        if m:
            case_ids.append(m.group(0))
        else:
            print("[WARNING] Cannot parse CASE_ID:", s)

elif subject_col is not None and study_col is not None and series_col is not None:
    print(f"Using columns: {subject_col}, {study_col}, {series_col}")

    for _, row in df.iterrows():
        if pd.isna(row[subject_col]) or pd.isna(row[study_col]) or pd.isna(row[series_col]):
            continue

        subject = str(row[subject_col]).strip()
        study = str(row[study_col]).strip()
        series = str(row[series_col]).strip()

        subject = str(int(float(subject))).zfill(5)
        study = str(int(float(study))).zfill(5)
        series = str(int(float(series)))

        case_ids.append(f"{subject}_{study}_{series}")

else:
    raise RuntimeError(
        "Cannot find columns. Need either CASE_ID column or SUBJECT_ID/STUDY_ID/SERIES_ID columns."
    )

case_ids = list(dict.fromkeys(case_ids))

with open(out, "w") as f:
    for cid in case_ids:
        f.write(cid + "\\n")

print(f"Saved {len(case_ids)} case IDs to {out}")
EOF

echo "========================================"
echo "Case list:"
cat "$CASE_LIST"
echo "========================================"

# =========================
# 2) Run TotalSegmentator phase prediction
# =========================
while read -r CASE_ID; do
    [ -z "$CASE_ID" ] && continue

    IFS="_" read -r SUBJECT_ID STUDY_ID SERIES_ID <<< "$CASE_ID"

    CT="${CT_ROOT}/${SUBJECT_ID}/${STUDY_ID}/${CASE_ID}.nii.gz"
    OUT_JSON="${OUT_DIR}/${CASE_ID}_phase.json"

    echo "========================================"
    echo "CASE_ID: $CASE_ID"
    echo "CT:      $CT"
    echo "OUT:     $OUT_JSON"

    if [ ! -f "$CT" ]; then
        echo "[SKIP] Missing CT: $CT"
        echo "$CASE_ID missing CT" >> "$MISSING_LIST"
        continue
    fi

    if [ -f "$OUT_JSON" ]; then
        echo "[SKIP] Output already exists: $OUT_JSON"
        continue
    fi

    if ! totalseg_get_phase -i "$CT" -o "$OUT_JSON"; then
        echo "[FAILED] $CASE_ID"
        echo "$CASE_ID" >> "$FAILED_LIST"
        continue
    fi

    echo "[DONE] $CASE_ID"

done < "$CASE_LIST"

# =========================
# 3) Merge all JSON outputs into one table
# =========================
echo "========================================"
echo "Merging phase JSON files into summary table..."

python - <<EOF
import os
import re
import json
import glob
import pandas as pd

out_dir = "$OUT_DIR"
summary_csv = "$SUMMARY_CSV"
summary_xlsx = "$SUMMARY_XLSX"

json_files = sorted(glob.glob(os.path.join(out_dir, "*_phase.json")))

def flatten_json(obj, prefix=""):
    """
    Flatten nested json safely.
    Dict -> dotted columns.
    List/scalar -> json string or scalar.
    """
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    elif isinstance(obj, list):
        # Keep lists readable in one cell
        out[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix] = obj

    return out

rows = []

for jf in json_files:
    base = os.path.basename(jf)
    case_id = base.replace("_phase.json", "")

    m = re.match(r"(\\d{5})_(\\d{5})_(\\d+)", case_id)
    if m:
        subject_id, study_id, series_id = m.groups()
    else:
        subject_id, study_id, series_id = None, None, None

    row = {
        "case_id": case_id,
        "subject_id": subject_id,
        "study_id": study_id,
        "series_id": series_id,
        "json_path": jf,
    }

    try:
        with open(jf, "r") as f:
            data = json.load(f)

        flat = flatten_json(data)
        row.update(flat)
        row["json_read_status"] = "ok"

    except Exception as e:
        row["json_read_status"] = "failed"
        row["json_error"] = str(e)

    rows.append(row)

df = pd.DataFrame(rows)

# Put useful ID columns first
first_cols = ["case_id", "subject_id", "study_id", "series_id", "json_read_status", "json_path"]
first_cols = [c for c in first_cols if c in df.columns]
other_cols = [c for c in df.columns if c not in first_cols]
df = df[first_cols + other_cols]

df.to_csv(summary_csv, index=False)

try:
    df.to_excel(summary_xlsx, index=False)
    print(f"Saved XLSX: {summary_xlsx}")
except Exception as e:
    print(f"[WARNING] Could not save XLSX: {e}")

print(f"Saved CSV:  {summary_csv}")
print(f"Total JSON files merged: {len(json_files)}")
print(f"Total rows: {len(df)}")
print("Columns:")
print(list(df.columns))
print("")
print("Preview:")
print(df.head())
EOF

echo "========================================"
echo "Finished time: $(date)"

echo "Output json count:"
find "$OUT_DIR" -maxdepth 1 -name "*_phase.json" | wc -l

echo "Summary CSV:"
echo "$SUMMARY_CSV"

echo "Summary XLSX:"
echo "$SUMMARY_XLSX"

echo "Missing CT cases:"
if [ -f "$MISSING_LIST" ]; then
    cat "$MISSING_LIST"
else
    echo "None"
fi

echo "Failed cases:"
if [ -f "$FAILED_LIST" ]; then
    cat "$FAILED_LIST"
else
    echo "None"
fi