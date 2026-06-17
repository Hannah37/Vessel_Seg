from pathlib import Path
import nibabel as nib
import numpy as np

SRC_ROOT = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/gt_sma_smv")
DST_ROOT = Path("/data/drdcad/Hyuna/projects/vessel_seg/data/gt_sma_smv_nnunet")

DST_ROOT.mkdir(parents=True, exist_ok=True)

for gt_path in sorted(SRC_ROOT.glob("*_sma_smv.nii.gz")):
    # macOS metadata 파일 무시
    if gt_path.name.startswith("._"):
        continue

    img = nib.load(str(gt_path))
    arr = np.asanyarray(img.dataobj)

    unique_before = np.unique(arr)

    # label 변환: SMV 3 -> 2
    arr_new = arr.copy()
    arr_new[arr_new == 3] = 2

    unique_after = np.unique(arr_new)

    # nnU-Net label은 integer로 저장
    arr_new = arr_new.astype(np.uint8)

    out_img = nib.Nifti1Image(arr_new, img.affine, img.header.copy())
    out_img.set_data_dtype(np.uint8)

    out_path = DST_ROOT / gt_path.name
    nib.save(out_img, str(out_path))

    print(f"{gt_path.name}")
    print(f"  before: {unique_before}")
    print(f"  after : {unique_after}")

print("\nDone.")