from pathlib import Path
import json
import shutil
import pandas as pd


# ============================================================
# Paths
# ============================================================
DATASET_ID = 101
DATASET_NAME = "Dataset101_SBOvessel"

IMG_ROOT = Path("/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis")
GT_ROOT = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/gt_vessel_algo")

# TODO: 엑셀 파일 경로만 네 실제 파일 위치에 맞게 확인
EXCEL_PATH = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/5samples_updated.xlsx")
SHEET_NAME = "used_data" 

NNUNET_RAW = Path("/data/drdcad/datasets_nnUNet/nnUNet_raw")
DATASET_DIR = NNUNET_RAW / DATASET_NAME


# ============================================================
# Label definition
# ============================================================
# gt_vessel_algo가 binary vessel label이면 이걸 사용
LABELS = {
    "background": 0,
    "vessel": 1,
}

# 만약 label이 sma=1, smv=2처럼 multi-class면 위 대신 아래처럼 바꿔야 함
# LABELS = {
#     "background": 0,
#     "sma": 1,
#     "smv": 2,
# }


# ============================================================
# Helpers
# ============================================================
def normalize_colname(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def clean_id(x, width=None):
    """
    Excel에서 4, 4.0, '00004' 등으로 읽혀도 case id 형식에 맞게 정리.
    subject/study는 width=5로 00004 형태로 맞춤.
    series는 보통 width=None.
    """
    if pd.isna(x):
        return None

    s = str(x).strip()

    # Excel에서 4.0처럼 들어오는 경우 처리
    if s.endswith(".0"):
        s = s[:-2]

    # 파일명 형태로 들어온 경우 확장자 제거
    s = s.replace(".nii.gz", "")
    s = s.replace("_0000", "")

    if width is not None:
        s = s.zfill(width)

    return s


def find_column(df, candidates):
    """
    후보 column 이름 중 실제 df에 존재하는 column을 찾음.
    """
    norm_to_original = {normalize_colname(c): c for c in df.columns}

    for cand in candidates:
        cand_norm = normalize_colname(cand)
        if cand_norm in norm_to_original:
            return norm_to_original[cand_norm]

    return None

def get_case_id_from_row(row, df):
    """
    Excel used_data 탭에서 case_id 생성.

    현재 used_data column:
        anon_patient_ID
        anon_study_ID
        SeriesNumber

    case_id 예:
        00004_00001_3
    """

    # SeriesNumber가 없으면 사용할 수 없는 row
    if "SeriesNumber" in df.columns and pd.isna(row["SeriesNumber"]):
        return None

    # 현재 Excel column 이름을 직접 사용
    if (
        "anon_patient_ID" in df.columns
        and "anon_study_ID" in df.columns
        and "SeriesNumber" in df.columns
    ):
        subject_id = clean_id(row["anon_patient_ID"], width=5)
        study_id = clean_id(row["anon_study_ID"], width=5)
        series_id = clean_id(row["SeriesNumber"], width=None)

        if subject_id is None or study_id is None or series_id is None:
            return None

        return f"{subject_id}_{study_id}_{series_id}"

    # 혹시 다른 column 이름으로 되어 있을 때 fallback
    case_col = find_column(
        df,
        [
            "case_id",
            "caseid",
            "case",
            "filename",
            "file_name",
            "image",
            "image_name",
            "nii",
            "nii_name",
        ],
    )

    if case_col is not None:
        case_id = clean_id(row[case_col])
        return case_id

    subject_col = find_column(
        df,
        [
            "anon_patient_ID",
            "anon_patient_id",
            "subject_id",
            "subject",
            "subj",
            "patient_id",
            "patient",
        ],
    )
    study_col = find_column(
        df,
        [
            "anon_study_ID",
            "anon_study_id",
            "study_id",
            "study",
            "studyid",
        ],
    )
    series_col = find_column(
        df,
        [
            "SeriesNumber",
            "series_number",
            "series_id",
            "series",
            "seriesid",
        ],
    )

    if subject_col is None or study_col is None or series_col is None:
        raise ValueError(
            "used_data 탭에서 case_id column 또는 "
            "anon_patient_ID/anon_study_ID/SeriesNumber column을 찾지 못했습니다.\n"
            f"현재 columns: {list(df.columns)}"
        )

    subject_id = clean_id(row[subject_col], width=5)
    study_id = clean_id(row[study_col], width=5)
    series_id = clean_id(row[series_col], width=None)

    if subject_id is None or study_id is None or series_id is None:
        return None

    return f"{subject_id}_{study_id}_{series_id}"

def get_split_from_row(row, df):
    """
    Excel에 split column이 있으면 train/test 구분.
    없으면 전부 train으로 처리.
    """
    split_col = find_column(
        df,
        [
            "split",
            "set",
            "train_test",
            "train_or_test",
            "dataset_split",
        ],
    )

    if split_col is None:
        return "train"

    value = str(row[split_col]).strip().lower()

    if value in ["test", "testing", "ts", "imagesTs"]:
        return "test"

    return "train"


def should_use_row(row, df):
    """
    used_data 탭에 used/include column이 있으면 false인 row는 제외.
    없으면 전부 사용.
    """
    use_col = find_column(df, ["used", "use", "include", "selected"])

    if use_col is None:
        return True

    value = str(row[use_col]).strip().lower()

    if value in ["0", "false", "no", "n", "exclude", "excluded"]:
        return False

    return True


def find_gt_path(case_id):
    """
    gt_vessel_algo 안에서 case_id에 해당하는 label 파일을 찾음.
    naming이 약간 달라도 찾을 수 있게 여러 pattern 사용.
    """
    candidates = [
        GT_ROOT / f"{case_id}.nii.gz",
        GT_ROOT / f"{case_id}_vessel.nii.gz",
        GT_ROOT / f"{case_id}_vessel_algo.nii.gz",
        GT_ROOT / f"{case_id}_gt.nii.gz",
        GT_ROOT / f"{case_id}_label.nii.gz",
    ]

    for p in candidates:
        if p.exists() and not p.name.startswith("._"):
            return p

    matches = sorted(
        [
            p for p in GT_ROOT.glob(f"{case_id}*.nii.gz")
            if not p.name.startswith("._")
        ]
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(f"WARNING: Multiple GT files found for {case_id}. Using first one:")
        for m in matches:
            print(f"  {m}")
        return matches[0]

    return None


# ============================================================
# Re-create nnU-Net raw dataset folder
# ============================================================
if DATASET_DIR.exists():
    print(f"Removing existing dataset folder: {DATASET_DIR}")
    shutil.rmtree(DATASET_DIR)

imagesTr = DATASET_DIR / "imagesTr"
labelsTr = DATASET_DIR / "labelsTr"
imagesTs = DATASET_DIR / "imagesTs"

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)
imagesTs.mkdir(parents=True, exist_ok=True)


# ============================================================
# Read Excel used_data sheet
# ============================================================
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, dtype=str)

print(f"Loaded Excel: {EXCEL_PATH}")
print(f"Sheet: {SHEET_NAME}")
print(f"Columns: {list(df.columns)}")
print(f"Rows: {len(df)}")


# ============================================================
# Link data
# ============================================================
linked_train_cases = []
linked_test_cases = []

missing_images = []
missing_labels = []
skipped_cases = []

seen_cases = set()

for _, row in df.iterrows():
    if not should_use_row(row, df):
        continue

    case_id = get_case_id_from_row(row, df)

    if case_id is None or case_id == "" or case_id.lower() == "nan":
        continue

    if case_id in seen_cases:
        print(f"WARNING: duplicated case_id in Excel, skipping duplicate: {case_id}")
        continue

    seen_cases.add(case_id)

    try:
        subject_id, study_id, series_id = case_id.split("_")
    except ValueError:
        print(f"WARNING: invalid case_id format, skipping: {case_id}")
        skipped_cases.append(case_id)
        continue

    img_path = IMG_ROOT / subject_id / study_id / f"{case_id}.nii.gz"

    if not img_path.exists():
        missing_images.append(str(img_path))
        continue

    split = get_split_from_row(row, df)

    if split == "test":
        img_link = imagesTs / f"{case_id}_0000.nii.gz"
        img_link.symlink_to(img_path)
        linked_test_cases.append(case_id)

    else:
        gt_path = find_gt_path(case_id)

        if gt_path is None:
            missing_labels.append(case_id)
            continue

        img_link = imagesTr / f"{case_id}_0000.nii.gz"
        lbl_link = labelsTr / f"{case_id}.nii.gz"

        img_link.symlink_to(img_path)
        lbl_link.symlink_to(gt_path)

        linked_train_cases.append(case_id)


# ============================================================
# Write dataset.json
# ============================================================
dataset_json = {
    "channel_names": {
        "0": "CT"
    },
    "labels": LABELS,
    "numTraining": len(linked_train_cases),
    "file_ending": ".nii.gz"
}

with open(DATASET_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)


# ============================================================
# Summary
# ============================================================
print("\n========================================")
print("Linked training cases")
print("========================================")
for c in linked_train_cases:
    print(c)

print("\n========================================")
print("Linked test cases")
print("========================================")
for c in linked_test_cases:
    print(c)

print("\n========================================")
print("Summary")
print("========================================")
print(f"Dataset dir: {DATASET_DIR}")
print(f"Training cases: {len(linked_train_cases)}")
print(f"Test cases: {len(linked_test_cases)}")
print(f"Missing images: {len(missing_images)}")
print(f"Missing labels: {len(missing_labels)}")
print(f"Skipped cases: {len(skipped_cases)}")

if missing_images:
    print("\nMissing images:")
    for m in missing_images:
        print(m)

if missing_labels:
    print("\nMissing labels:")
    for c in missing_labels:
        print(c)

if skipped_cases:
    print("\nSkipped cases:")
    for c in skipped_cases:
        print(c)

print("\nDone.")