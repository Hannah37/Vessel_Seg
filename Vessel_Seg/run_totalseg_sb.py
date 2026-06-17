import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


# =========================
# Paths
# =========================
EXCEL_PATH = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/5samples_updated.xlsx")

IN_ROOT = Path(
    "/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis"
)

OUT_DIR = Path(
    "/data/drdcad/Hyuna/projects/vessel_seg/data/totalseg_output"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Load used_data sheet
# =========================
df = pd.read_excel(EXCEL_PATH, sheet_name="used_data") #### 유효한 데이터만 사용

required_cols = ["anon_patient_ID", "anon_study_ID", "SeriesNumber"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column in used_data sheet: {col}")

df = df.dropna(subset=required_cols)

print(f"Number of rows in used_data: {len(df)}")


# =========================
# Run TotalSegmentator
# =========================
for _, row in df.iterrows():
    patient_id = int(row["anon_patient_ID"])
    study_id = int(row["anon_study_ID"])
    series_num = int(row["SeriesNumber"])

    case_id = f"{patient_id:05d}_{study_id:05d}_{series_num}"

    input_nii = (
        IN_ROOT
        / f"{patient_id:05d}"
        / f"{study_id:05d}"
        / f"{case_id}.nii.gz"
    )

    final_output = OUT_DIR / f"{case_id}_small_bowel.nii.gz"
    tmp_dir = OUT_DIR / f"tmp_{case_id}"

    print("=" * 80)
    print(f"Case:   {case_id}")
    print(f"Input:  {input_nii}")
    print(f"Output: {final_output}")

    if not input_nii.exists():
        print(f"[SKIP] Input file does not exist: {input_nii}")
        continue

    if final_output.exists():
        print(f"[SKIP] Output already exists: {final_output}")
        continue

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "TotalSegmentator",
        "-i", str(input_nii),
        "-o", str(tmp_dir),
        "-rs", "small_bowel",
    ]

    print("[RUN]", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] TotalSegmentator failed for {case_id}")
        print(e)
        continue

    small_bowel_file = tmp_dir / "small_bowel.nii.gz"

    if not small_bowel_file.exists():
        print(f"[ERROR] small_bowel.nii.gz was not created for {case_id}")
        continue

    shutil.move(str(small_bowel_file), str(final_output))
    shutil.rmtree(tmp_dir)

    print(f"[SAVED] {final_output}")


# =========================
# Check number of output files
# =========================
output_files = sorted(OUT_DIR.glob("*_small_bowel.nii.gz"))

print("=" * 80)
print(f"Number of saved small bowel files: {len(output_files)}")
print(f"Expected number of cases from used_data: {len(df)}")

if len(output_files) == len(df):
    print("[OK] Output file count matches used_data rows.")
else:
    print("[WARNING] Output file count does not match used_data rows.")
    
print("Done.")