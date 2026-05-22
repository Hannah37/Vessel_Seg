import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.morphology import ball, binary_dilation, remove_small_objects
from skimage.filters import frangi
import torch
import torch.nn.functional as F
import time
from skimage.morphology import skeletonize_3d

def load_nifti(path):
    nii = nib.load(path)
    arr = np.asanyarray(nii.dataobj)
    return arr, nii.affine, nii.header


def save_nifti(arr, affine, header, out_path):
    out = nib.Nifti1Image(arr.astype(np.uint8), affine, header)
    nib.save(out, out_path)


def crop_bbox(mask, margin=80):
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Input mask is empty.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, mask.shape)

    return tuple(slice(mins[i], maxs[i]) for i in range(3))


def keep_connected_to_seed(candidate, seed):
    structure = ndi.generate_binary_structure(3, 2)
    labeled, _ = ndi.label(candidate, structure=structure)

    seed_labels = np.unique(labeled[seed])
    seed_labels = seed_labels[seed_labels != 0]

    if len(seed_labels) == 0:
        return np.zeros_like(candidate, dtype=bool)

    return np.isin(labeled, seed_labels)

def make_vessel_candidate(
    ct_crop,
    sma_valid,
    vein_valid,
    hu_min=50,
    hu_max=450,
    vesselness_thr=0.003,
    use_frangi=True,
    territory_ratio=1.05,
    dilate_radius=1,
):
    ct_f = ct_crop.astype(np.float32)

    candidate = (ct_f >= hu_min) & (ct_f <= hu_max)

    if use_frangi:
        print("Computing Frangi vesselness...")
        vness = frangi(
            ct_f,
            sigmas=[0.3, 0.5, 1, 2],
            black_ridges=False,
        )
        candidate = candidate & (vness >= vesselness_thr)

    dist_to_sma = ndi.distance_transform_edt(~sma_valid)
    dist_to_vein = ndi.distance_transform_edt(~vein_valid)

    artery_territory = dist_to_sma < (dist_to_vein * territory_ratio)

    candidate = candidate & artery_territory

    # 끊긴 작은 artery 연결용: 아주 약하게만
    if dilate_radius > 0:
        candidate = binary_dilation(candidate, ball(1))

        candidate = ndi.binary_closing(
            candidate,
            structure=ball(1)
        )

        candidate = candidate & artery_territory
        candidate = candidate | sma_valid

    # seed는 반드시 포함
    candidate = candidate | sma_valid

    return candidate, artery_territory

def competitive_grow_gpu(candidate, sma_seed, vein_seed, max_iter=800, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cand = torch.from_numpy(candidate.astype(np.bool_)).to(device)
    sma = torch.from_numpy(sma_seed.astype(np.bool_)).to(device) & cand
    vein = torch.from_numpy(vein_seed.astype(np.bool_)).to(device) & cand

    # seed는 반드시 포함
    cand = cand | sma | vein

    kernel = torch.ones((1,1,3,3,3), device=device)

    # corner 제외
    kernel[0,0,0,0,0] = 0
    kernel[0,0,0,0,2] = 0
    kernel[0,0,0,2,0] = 0
    kernel[0,0,0,2,2] = 0
    kernel[0,0,2,0,0] = 0
    kernel[0,0,2,0,2] = 0
    kernel[0,0,2,2,0] = 0
    kernel[0,0,2,2,2] = 0

    kernel[0,0,1,1,1] = 0

    sma_cur = sma[None, None]
    vein_cur = vein[None, None]
    sma_front = sma_cur.clone()
    vein_front = vein_cur.clone()
    cand = cand[None, None]

    for i in range(max_iter):
        occupied = sma_cur | vein_cur

        sma_grown = F.conv3d(sma_front.float(), kernel, padding=1) > 0
        vein_grown = F.conv3d(vein_front.float(), kernel, padding=1) > 0

        sma_new = sma_grown & cand & (~occupied)
        vein_new = vein_grown & cand & (~occupied)

        # 같은 iteration에서 둘 다 도달한 voxel은 ambiguous → 제거
        conflict = sma_new & vein_new
        sma_new = sma_new & (~conflict)
        vein_new = vein_new & (~conflict)

        n_new = int(sma_new.sum().item() + vein_new.sum().item())
        if n_new == 0:
            print(f"Converged at iteration {i}")
            break

        sma_cur |= sma_new
        vein_cur |= vein_new

        sma_front = sma_new
        vein_front = vein_new

        if i % 20 == 0:
            print(
                f"\rIter {i} | SMA: {int(sma_cur.sum())} | SMV: {int(vein_cur.sum())}",
                end="",
                flush=True
            )

    print(flush=True)
    return sma_cur.squeeze().detach().cpu().numpy().astype(bool), vein_cur.squeeze().detach().cpu().numpy().astype(bool)

def filter_by_sma_cross_section_area(vessel, sma, axis=2, scale=1.5):
    sma_areas = []
    for z in range(sma.shape[axis]):
        sma_slice = np.take(sma, z, axis=axis)
        sma_areas.append(sma_slice.sum())

    max_sma_area = max(sma_areas)
    max_allowed_area = max_sma_area * scale

    filtered = np.zeros_like(vessel, dtype=bool)
    structure2d = ndi.generate_binary_structure(2, 2)

    for z in range(vessel.shape[axis]):
        vessel_slice = np.take(vessel, z, axis=axis)
        labeled, n = ndi.label(vessel_slice, structure=structure2d)

        for lab in range(1, n + 1):
            comp = labeled == lab
            if comp.sum() <= max_allowed_area:
                filtered[:, :, z] |= comp

    return filtered

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ct",
        default="/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis/00002/00001/00002_00001_3.nii.gz",
    )
    parser.add_argument("--seg", default="sma_smv.nii.gz")
    parser.add_argument("--out", default="branches_tree.nii.gz")

    parser.add_argument("--hu_min", type=float, default=80)
    parser.add_argument("--hu_max", type=float, default=350)
    parser.add_argument("--margin", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=800)
    parser.add_argument("--min_size", type=int, default=3)

    parser.add_argument("--vesselness_thr", type=float, default=0.02)
    parser.add_argument("--max_radius_vox", type=int, default=6)
    parser.add_argument("--vein_block_radius", type=int, default=3)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    ct, affine, header = load_nifti(args.ct)
    sma_smv, _, _ = load_nifti(args.seg)

    sma = sma_smv == 1
    vein = sma_smv == 3

    roi_mask = sma | vein
    slices = crop_bbox(roi_mask, margin=args.margin)

    ct_crop = ct[slices]
    sma_crop = sma[slices]
    vein_crop = vein[slices]

    sma_valid = sma_crop & (ct_crop >= 0)
    vein_valid = vein_crop & (ct_crop >= 0)

    print("Artery seed voxels:", sma_valid.sum())
    print("Vein seed voxels:", vein_valid.sum())

    candidate, artery_territory = make_vessel_candidate(
        ct_crop=ct_crop,
        sma_valid=sma_valid,
        vein_valid=vein_valid,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        vesselness_thr=args.vesselness_thr,
        use_frangi=True,
        territory_ratio=1.5,
        dilate_radius=1,
    )

    print("Candidate voxels:", candidate.sum())

    vein_exclude = binary_dilation(
        vein_valid,
        ball(args.vein_block_radius),
    )

    # Never remove SMA seed itself
    vein_exclude = vein_exclude & (~sma_valid)

    print("Vein-exclude voxels:", vein_exclude.sum())

    # 2. Grow SMA-connected arterial tree while excluding vein tree
    print("Growing SMA arterial tree...")
        
    artery_tree, vein_tree = competitive_grow_gpu(
    candidate=candidate,
    sma_seed=sma_valid,
    vein_seed=vein_valid,
    max_iter=args.max_iter,
    device=args.device,
)

    # 1) 첫 번째 코드처럼 slice-wise 큰 organ/blob 제거
    artery_tree = filter_by_sma_cross_section_area(
        vessel=artery_tree,
        sma=sma_valid,
        axis=2,
        scale=1.2,
    )

    # 3) SMA와 연결된 tree만 유지
    artery_tree = keep_connected_to_seed(
    candidate=artery_tree | sma_valid,
    seed=sma_valid
)

    artery_tree = artery_tree & (~vein_tree)

    artery_tree = remove_small_objects(
        artery_tree.astype(bool),
        min_size=10
    )

    # topology refinement
    skeleton = skeletonize_3d(artery_tree)

    artery_tree = binary_dilation(
        skeleton,
        ball(1)
    )

    artery_tree = keep_connected_to_seed(
        candidate=artery_tree | sma_valid,
        seed=sma_valid
    )

    artery_tree = artery_tree | sma_valid   
        
    result = np.zeros(sma.shape, dtype=np.uint8)
    result[slices] = artery_tree.astype(np.uint8)

    save_nifti(result, affine, header, args.out)

    print("Saved:", args.out)
    print("Output voxels:", result.sum())

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Total runtime: {minutes} min {seconds:.2f} sec")

if __name__ == "__main__":
    main()
