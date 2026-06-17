from pathlib import Path
import pandas as pd

DATASET_NAME = "Dataset100_SmallBowelObstruction"

IMG_ROOT = Path("/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis")
EXCEL_PATH = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/5samples_updated.xlsx")

DATASET_DIR = Path("/data/drdcad/datasets_nnUNet/nnUNet_raw") / DATASET_NAME

imagesTr = DATASET_DIR / "imagesTr"
labelsTr = DATASET_DIR / "labelsTr"
imagesTs = DATASET_DIR / "imagesTs"

imagesTs.mkdir(parents=True, exist_ok=True)

# 기존 imagesTs 비우기
for p in imagesTs.glob("*.nii.gz"):
    p.unlink()

# 현재 training에 쓰인 case 읽기
# labelsTr filename: 00001_00001_3.nii.gz
used_cases = set()

for p in labelsTr.glob("*.nii.gz"):
    if p.name.startswith("._"):
        continue
    case_id = p.name.replace(".nii.gz", "")
    used_cases.add(case_id)

train_count = len(used_cases)

print(f"Train data count from labelsTr: {train_count}")
print(sorted(used_cases))

# Excel 읽기
df_all = pd.read_excel(EXCEL_PATH, sheet_name="used_data")

# used_data 탭 전체 row 개수
used_data_total_count = len(df_all)

# SeriesNumber 있는 row만 사용
df = df_all[df_all["SeriesNumber"].notna()].copy()

# 실제 test 후보로 사용할 수 있는 row 개수
used_data_with_series_count = len(df)

print(f"\nUsed_data total count: {used_data_total_count}")
print(f"Used_data count with SeriesNumber: {used_data_with_series_count}")

num_test = 0
missing = []
linked_cases = []

for _, row in df.iterrows():
    subject_id = f"{int(row['anon_patient_ID']):05d}"
    study_id = f"{int(row['anon_study_ID']):05d}"
    series_id = str(int(row["SeriesNumber"]))

    case_id = f"{subject_id}_{study_id}_{series_id}"

    # training에 쓰인 case 제외
    if case_id in used_cases:
        continue

    img_path = IMG_ROOT / subject_id / study_id / f"{case_id}.nii.gz"

    if not img_path.exists():
        missing.append(str(img_path))
        continue

    img_link = imagesTs / f"{case_id}_0000.nii.gz"

    if img_link.exists() or img_link.is_symlink():
        img_link.unlink()

    img_link.symlink_to(img_path)

    num_test += 1
    linked_cases.append(case_id)

print(f"\nPrediction cases linked: {num_test}")

for c in linked_cases:
    print(f"Linked test: {c}")

if missing:
    print("\nMissing files:")
    for m in missing:
        print(m)

# 마지막에 실제 imagesTs 안의 test 파일 개수 확인
actual_test_files = sorted([
    p for p in imagesTs.glob("*.nii.gz")
    if not p.name.startswith("._")
])

print("\n==============================")
print(f"Used_data total count: {used_data_total_count}")
print(f"Used_data count with SeriesNumber: {used_data_with_series_count}")
print(f"Train data count from labelsTr: {train_count}")
print(f"Prediction cases linked this run: {num_test}")
print(f"Final test data count in imagesTs: {len(actual_test_files)}")
print(f"Missing source image count: {len(missing)}")
print("==============================")