from pathlib import Path
import json
import random
import pandas as pd
from collections import defaultdict


BASE_DIR = Path("/vf/users/drdcad/Hyuna/projects/vessel_seg/code/train_DiffUNet")

IMG_ROOT = Path("/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis")
GT_ROOT = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/gt_vessel_algo")
EXCEL_PATH = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/5samples_updated.xlsx")

OUT_JSON_DIR = BASE_DIR / "jsons"
OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_JSON_DIR / "sbo_vessel_5fold.json"

SHEET_NAME = "used_data"
N_FOLDS = 5
SEED = 42


def clean_id(x, width=None):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    s = s.replace(".nii.gz", "")
    s = s.replace("_0000", "")

    if width is not None:
        s = s.zfill(width)

    return s


def get_case_id_from_row(row):
    if pd.isna(row["SeriesNumber"]):
        return None

    subject_id = clean_id(row["anon_patient_ID"], width=5)
    study_id = clean_id(row["anon_study_ID"], width=5)
    series_id = clean_id(row["SeriesNumber"])

    if subject_id is None or study_id is None or series_id is None:
        return None

    return f"{subject_id}_{study_id}_{series_id}"


def extract_case_id_from_gt(gt_path):
    if gt_path.name.startswith("._"):
        return None

    if not gt_path.name.endswith(".nii.gz"):
        return None

    stem = gt_path.name.replace(".nii.gz", "")

    suffixes = [
        "_vessel_algo",
        "_vessel",
        "_label",
        "_gt",
        "_sma_smv",
    ]

    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return stem


def collect_gt_dict():
    gt_dict = {}

    for gt_path in sorted(GT_ROOT.glob("*.nii.gz")):
        case_id = extract_case_id_from_gt(gt_path)

        if case_id is None:
            continue

        if case_id in gt_dict:
            raise RuntimeError(
                f"Duplicate GT for case_id={case_id}\n"
                f"1) {gt_dict[case_id]}\n"
                f"2) {gt_path}"
            )

        gt_dict[case_id] = gt_path

    return gt_dict


df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

used_cases = []

for _, row in df.iterrows():
    case_id = get_case_id_from_row(row)

    if case_id is None:
        continue

    used_cases.append(case_id)

used_case_set = set(used_cases)

if len(used_cases) != len(used_case_set):
    duplicated = sorted([c for c in used_case_set if used_cases.count(c) > 1])
    raise RuntimeError(
        "Duplicate case_id found in used_data:\n" + "\n".join(duplicated)
    )

print(f"used_data valid cases: {len(used_case_set)}")

gt_dict = collect_gt_dict()
gt_case_set = set(gt_dict.keys())

print(f"gt_vessel_algo cases: {len(gt_case_set)}")

missing_gt = sorted(used_case_set - gt_case_set)
extra_gt = sorted(gt_case_set - used_case_set)

if missing_gt:
    print("\nERROR: In used_data but missing in gt_vessel_algo:")
    for c in missing_gt:
        print(c)

if extra_gt:
    print("\nERROR: In gt_vessel_algo but not in used_data:")
    for c in extra_gt:
        print(c)

if missing_gt or extra_gt:
    raise RuntimeError(
        "used_data case set and gt_vessel_algo case set do not match exactly."
    )

print("OK: used_data and gt_vessel_algo match exactly.")

items = []
missing_images = []

for case_id in sorted(used_case_set):
    subject_id, study_id, series_id = case_id.split("_")

    image_path = IMG_ROOT / subject_id / study_id / f"{case_id}.nii.gz"
    label_path = gt_dict[case_id]

    if not image_path.exists():
        missing_images.append(str(image_path))
        continue

    if not label_path.exists():
        raise RuntimeError(f"Missing label path: {label_path}")

    items.append(
        {
            "case_id": case_id,
            "subject_id": subject_id,
            "image": str(image_path),
            "label": str(label_path),
        }
    )

if missing_images:
    print("\nERROR: Missing images:")
    for p in missing_images:
        print(p)
    raise RuntimeError("Some CT images are missing.")

print(f"Final usable items: {len(items)}")

# subject-level split
subject_to_items = defaultdict(list)

for item in items:
    subject_to_items[item["subject_id"]].append(item)

subjects = sorted(subject_to_items.keys())

random.seed(SEED)
random.shuffle(subjects)

fold_subjects = [[] for _ in range(N_FOLDS)]

for i, subject_id in enumerate(subjects):
    fold_subjects[i % N_FOLDS].append(subject_id)

out = {
    "description": "SBO vessel segmentation 5-fold split for Diff-UNet",
    "labels": {
        "0": "background",
        "1": "vessel",
    },
    "num_cases": len(items),
    "num_subjects": len(subjects),
    "folds": {},
}

for fold in range(N_FOLDS):
    val_subjects = set(fold_subjects[fold])

    train_items = []
    val_items = []

    for subject_id, subject_items in subject_to_items.items():
        if subject_id in val_subjects:
            val_items.extend(subject_items)
        else:
            train_items.extend(subject_items)

    train_items = sorted(train_items, key=lambda x: x["case_id"])
    val_items = sorted(val_items, key=lambda x: x["case_id"])

    train_subjects = {x["subject_id"] for x in train_items}
    val_subjects_check = {x["subject_id"] for x in val_items}

    overlap = train_subjects & val_subjects_check
    if overlap:
        raise RuntimeError(f"Subject leakage in fold {fold}: {sorted(overlap)}")

    out["folds"][str(fold)] = {
        "training": train_items,
        "validation": val_items,
    }

    print(
        f"Fold {fold}: "
        f"train={len(train_items)}, val={len(val_items)}, "
        f"train_subjects={len(train_subjects)}, val_subjects={len(val_subjects_check)}"
    )

with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=4)

print(f"\nSaved: {OUT_JSON}")