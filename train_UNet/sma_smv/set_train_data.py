from pathlib import Path
import json
import shutil

DATASET_NAME = "Dataset100_SmallBowelObstruction"

IMG_ROOT = Path("/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis")
GT_ROOT = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/gt_sma_smv_nnunet")

NNUNET_RAW = Path("/data/drdcad/datasets_nnUNet/nnUNet_raw")
DATASET_DIR = NNUNET_RAW / DATASET_NAME

# Dataset100 raw 폴더 완전히 새로 만들기
if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)

imagesTr = DATASET_DIR / "imagesTr"
labelsTr = DATASET_DIR / "labelsTr"
imagesTs = DATASET_DIR / "imagesTs"

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)
imagesTs.mkdir(parents=True, exist_ok=True)

num_train = 0
linked_cases = []
missing_images = []

# GT 파일 기준으로 training case 생성
for gt_path in sorted(GT_ROOT.glob("*_sma_smv.nii.gz")):
    if gt_path.name.startswith("._"):
        continue

    # 예: 00004_00001_3_sma_smv.nii.gz -> 00004_00001_3
    case_id = gt_path.name.replace("_sma_smv.nii.gz", "")

    subject_id, study_id, series_id = case_id.split("_")

    img_path = IMG_ROOT / subject_id / study_id / f"{case_id}.nii.gz"

    if not img_path.exists():
        missing_images.append(str(img_path))
        continue

    img_link = imagesTr / f"{case_id}_0000.nii.gz"
    lbl_link = labelsTr / f"{case_id}.nii.gz"

    img_link.symlink_to(img_path)
    lbl_link.symlink_to(gt_path)

    linked_cases.append(case_id)
    num_train += 1

dataset_json = {
    "channel_names": {
        "0": "CT"
    },
    "labels": {
        "background": 0,
        "sma": 1,
        "smv": 2
    },
    "numTraining": num_train,
    "file_ending": ".nii.gz"
}

with open(DATASET_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print("\nLinked training cases:")
for c in linked_cases:
    print(c)

print(f"\nTraining cases: {num_train}")

if missing_images:
    print("\nMissing images:")
    for m in missing_images:
        print(m)