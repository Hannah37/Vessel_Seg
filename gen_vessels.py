import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.morphology import ball, binary_dilation, remove_small_objects
import torch
import torch.nn.functional as F

def load_nifti(path):
    nii = nib.load(path)
    arr = np.asanyarray(nii.dataobj)
    return arr, nii.affine, nii.header

def save_nifti(arr, affine, header, out_path):
    out = nib.Nifti1Image(arr.astype(np.uint8), affine, header)
    nib.save(out, out_path)

def crop_bbox(mask, margin=40):
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("SMA mask is empty.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, mask.shape)

    return tuple(slice(mins[i], maxs[i]) for i in range(3))

def region_grow_from_seed(
    ct,
    seed,
    hu_min=80,
    hu_max=500,
    max_iter=200,
    max_distance_mm=None,
    spacing=(1,1,1),
    exclude_mask=None,
):
    """
    Fast frontier-based region growing.
    """

    candidate = (ct >= hu_min) & (ct <= hu_max)

    if exclude_mask is not None:
        candidate &= (~exclude_mask)

    current = seed.copy()

    if exclude_mask is not None:
        current &= (~exclude_mask)

    if max_distance_mm is not None:
        dist = ndi.distance_transform_edt(~seed, sampling=spacing)
        candidate &= (dist <= max_distance_mm)

    structure = ndi.generate_binary_structure(3, 1)

    # frontier = 새로 자라날 boundary만 유지
    frontier = current.copy()

    for i in range(max_iter):

        # frontier만 dilation
        grown = ndi.binary_dilation(frontier, structure=structure)

        new_voxels = grown & candidate & (~current)

        if new_voxels.sum() == 0:
            print(f"Converged at iteration {i}")
            break

        current |= new_voxels

        # 다음 iteration에서는 새 voxel 주변만 탐색
        frontier = new_voxels

    return current

def filter_by_sma_cross_section_area(vessel, sma, axis=2, scale=1.0):
    """
    각 slice에서 component 단면적이 SMA의 최대 단면적보다 크면 제거.
    axis=2: axial slice 기준
    scale=1.0이면 SMA 최대 단면적 이하만 허용.
    """

    sma_areas = []
    for z in range(sma.shape[axis]):
        sma_slice = np.take(sma, z, axis=axis)
        sma_areas.append(sma_slice.sum())

    max_sma_area = max(sma_areas)

    if max_sma_area == 0:
        raise ValueError("SMA area is zero.")

    max_allowed_area = max_sma_area * scale
    filtered = np.zeros_like(vessel, dtype=bool)

    structure2d = ndi.generate_binary_structure(2, 2)

    for z in range(vessel.shape[axis]):
        vessel_slice = np.take(vessel, z, axis=axis)

        labeled, n = ndi.label(vessel_slice, structure=structure2d)

        for lab in range(1, n + 1):
            comp = labeled == lab
            area = comp.sum()

            if area <= max_allowed_area:
                if axis == 2:
                    filtered[:, :, z] |= comp
                elif axis == 1:
                    filtered[:, z, :] |= comp
                elif axis == 0:
                    filtered[z, :, :] |= comp

    print(f"Max SMA cross-section area: {max_sma_area}")
    print(f"Max allowed vessel area: {max_allowed_area}")

    return filtered

def filter_by_local_radius(mask, max_radius_vox=4):
    """
    너무 두꺼운 vessel/장기/vein 후보 제거.
    작은 artery branch는 유지.
    """
    dist = ndi.distance_transform_edt(mask)
    return mask & (dist <= max_radius_vox)

def keep_largest_connected(mask):
    structure = ndi.generate_binary_structure(3, 2)
    labeled, n = ndi.label(mask, structure=structure)

    if n == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()

    return labeled == largest

def keep_connected_to_seed(candidate, seed):
    """
    candidate 중에서 seed와 연결된 component만 유지.
    """
    structure = ndi.generate_binary_structure(3, 2)

    labeled, n = ndi.label(candidate, structure=structure)

    seed_labels = np.unique(labeled[seed])
    seed_labels = seed_labels[seed_labels != 0]

    if len(seed_labels) == 0:
        return np.zeros_like(candidate, dtype=bool)

    return np.isin(labeled, seed_labels)

def bridge_sma_to_vessel(vessel, sma, ct, radius=2):
    """
    SMA와 새 vessel 사이의 작은 gap을 ct >= 0 영역에서만 연결.
    """
    valid = ct >= 0
    selem = ball(radius)

    sma_grown = binary_dilation(sma, selem) & valid
    vessel_grown = binary_dilation(vessel, selem) & valid

    bridge = (sma_grown & vessel_grown) | sma_grown

    return (vessel | bridge | sma) & valid




def region_grow_tubular_gpu(
    ct,
    seed,
    exclude_mask=None,
    hu_min=80,
    hu_max=400,
    max_iter=150,
    device="cuda",
):
    """
    GPU accelerated tubular region growing.
    """

    ct_t = torch.from_numpy(ct).float().to(device)

    seed_t = torch.from_numpy(seed.astype(np.float32)).to(device)

    candidate = (ct_t >= hu_min) & (ct_t <= hu_max)

    if exclude_mask is not None:
        exclude_t = torch.from_numpy(exclude_mask.astype(np.bool_)).to(device)
        candidate &= (~exclude_t)

    current = seed_t.bool()

    # 6-neighborhood kernel
    kernel = torch.zeros((1,1,3,3,3), device=device)

    kernel[0,0,1,1,0] = 1
    kernel[0,0,1,1,2] = 1
    kernel[0,0,1,0,1] = 1
    kernel[0,0,1,2,1] = 1
    kernel[0,0,0,1,1] = 1
    kernel[0,0,2,1,1] = 1

    frontier = current.clone()

    current = current.unsqueeze(0).unsqueeze(0)
    frontier = frontier.unsqueeze(0).unsqueeze(0)
    candidate = candidate.unsqueeze(0).unsqueeze(0)

    for i in range(max_iter):

        grown = F.conv3d(
            frontier.float(),
            kernel,
            padding=1
        ) > 0

        new_voxels = grown & candidate & (~current)

        n_new = new_voxels.sum()

        if n_new == 0:
            print(f"Converged at iteration {i}")
            break

        current |= new_voxels

        frontier = new_voxels

    result = current.squeeze().detach().cpu().numpy()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ct",
        help="CT nifti path",
        default="/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis/00002/00001/00002_00001_3.nii.gz"
    )

    parser.add_argument(
        "--seg",
        help="SMA & SMV segmentation nifti path",
        default="sma_smv.nii.gz"
    )

    parser.add_argument(
        "--out",
        help="output nifti path",
        default="branches_cpu.nii.gz"
    )

    parser.add_argument("--hu_min", type=float, default=120)
    parser.add_argument("--hu_max", type=float, default=300)
    parser.add_argument("--margin", type=int, default=120)
    parser.add_argument("--max_iter", type=int, default=4000)
    parser.add_argument("--grow_radius", type=int, default=1)
    parser.add_argument("--max_distance_mm", type=float, default=300)
    parser.add_argument("--min_size", type=int, default=2)
    parser.add_argument("--vein_block_radius", type=int, default=3)

    args = parser.parse_args()
    ct, affine, header = load_nifti(args.ct)
    sma_smv, _, _ = load_nifti(args.seg)

    sma = sma_smv == 1
    vein = sma_smv == 3

    zooms = header.get_zooms()[:3]

    roi_mask = sma | vein
    slices = crop_bbox(roi_mask, margin=args.margin)

    ct_crop = ct[slices]
    sma_crop = sma[slices]
    vein_crop = vein[slices]

    sma_valid_crop = sma_crop & (ct_crop >= 0)
    vein_valid_crop = vein_crop & (ct_crop >= 0)
    vein_block_crop = binary_dilation(
        vein_valid_crop,
        ball(3)
    )

    vein_exclude_crop = vein_block_crop & (~sma_valid_crop)

    vessel_crop = region_grow_tubular_gpu(
        ct=ct_crop,
        seed=sma_valid_crop,
        exclude_mask=vein_exclude_crop,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        max_iter=args.max_iter,
    )

    vessel_crop = filter_by_sma_cross_section_area(
        vessel=vessel_crop,
        sma=sma_valid_crop,
        axis=2,
        scale=2.5
    )

    # 일단 비활성화 추천
    # vessel_crop = filter_by_local_radius(...)

    vessel_crop = bridge_sma_to_vessel(
        vessel=vessel_crop,
        sma=sma_valid_crop,
        ct=ct_crop,
        radius=2
    )

    vessel_crop = remove_small_objects(
        vessel_crop.astype(bool),
        min_size=3
    )

    vessel_crop = keep_connected_to_seed(
        candidate=vessel_crop | sma_valid_crop,
        seed=sma_valid_crop
    )

    vessel_crop = remove_small_objects(
        vessel_crop.astype(bool),
        min_size=3
    )

    vessel_crop = vessel_crop | sma_valid_crop

    result = np.zeros(sma.shape, dtype=np.uint8)
    result[slices] = vessel_crop.astype(np.uint8)

    save_nifti(result, affine, header, args.out)        

    print("Saved:", args.out)
    print("SMA voxels:", sma.sum())
    print("Output vessel voxels:", result.sum())


if __name__ == "__main__":
    main()