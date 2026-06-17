import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd

# =========================
# 설정
# =========================
PRED_DIR = "/data/drdcad/Hyuna/projects/vessel_seg/data/pred_sma_smv"

SMA_LABEL = 1
SMV_LABEL = 2   # 만약 SMV label이 3이면 3으로 변경

# =========================
# 파일 검색
# =========================
nii_files = sorted(
    glob.glob(os.path.join(PRED_DIR, "*.nii")) +
    glob.glob(os.path.join(PRED_DIR, "*.nii.gz"))
)

results = []

for f in nii_files:
    img = nib.load(f)
    seg = img.get_fdata()

    # voxel spacing 정보
    voxel_spacing = img.header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_spacing)

    # voxel count
    sma_voxels = int(np.sum(seg == SMA_LABEL))
    smv_voxels = int(np.sum(seg == SMV_LABEL))
    total_segmented_voxels = int(np.sum(seg > 0))

    # physical volume
    sma_volume_mm3 = sma_voxels * voxel_volume_mm3
    smv_volume_mm3 = smv_voxels * voxel_volume_mm3
    total_volume_mm3 = total_segmented_voxels * voxel_volume_mm3

    results.append({
        "filename": os.path.basename(f),
        "sma_voxels": sma_voxels,
        "smv_voxels": smv_voxels,
        "total_segmented_voxels": total_segmented_voxels,
        "sma_volume_mm3": sma_volume_mm3,
        "smv_volume_mm3": smv_volume_mm3,
        "total_volume_mm3": total_volume_mm3,
    })

# =========================
# DataFrame 생성 및 정렬
# =========================
df = pd.DataFrame(results)

# total segmented voxel 수 기준 오름차순 정렬
df_sorted = df.sort_values(by="total_segmented_voxels", ascending=True)

print(df_sorted.to_string(index=False))

# CSV로 저장하고 싶으면
df_sorted.to_csv("pred_sma_smv_sorted.csv", index=False)
print("\nSaved: pred_sma_smv_sorted.csv")