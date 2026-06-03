import argparse
import time
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import frangi
from skimage.graph import route_through_array
from skimage.morphology import ball, binary_dilation, remove_small_objects, skeletonize_3d


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
    # 예전 base처럼 18-connectivity 사용
    structure = ndi.generate_binary_structure(3, 2)
    labeled, _ = ndi.label(candidate.astype(bool), structure=structure)
    seed_labels = np.unique(labeled[seed.astype(bool)])
    seed_labels = seed_labels[seed_labels != 0]
    if len(seed_labels) == 0:
        return np.zeros_like(candidate, dtype=bool)
    return np.isin(labeled, seed_labels)

def make_vessel_candidate(
    ct_crop,
    sma_valid,
    vein_valid,
    hu_min=95,
    hu_max=370,
    vesselness_thr=0.0012,
    use_frangi=True,
    territory_ratio=1.8,
    dilate_radius=1,
    air_hu=-500,
    air_radius=2,
):
    ct_f = ct_crop.astype(np.float32)

    candidate_core = (ct_f >= hu_min) & (ct_f <= hu_max)

    if use_frangi:
        print("Computing Frangi vesselness...")
        vness = frangi(
            ct_f,
            sigmas=[0.2, 0.3, 0.5, 0.8, 1.2],
            black_ridges=False,
        )
        candidate_core = candidate_core & (vness >= vesselness_thr)

    dist_to_sma = ndi.distance_transform_edt(~sma_valid)
    dist_to_vein = ndi.distance_transform_edt(~vein_valid)

    artery_territory = dist_to_sma < (dist_to_vein * territory_ratio)
    candidate_core = candidate_core & artery_territory

    # Remove only long bowel-boundary-like components near air, not all bowel-adjacent vessels
    candidate_core = remove_bowel_boundary_like_components(
        mask=candidate_core,
        ct_crop=ct_crop,
        protected=sma_valid,
        air_hu=air_hu,
        air_radius=air_radius,
        min_area=20,
        min_long_axis=12,
        min_aspect=2.5,
        axis=2,
    )

    candidate_core = candidate_core | sma_valid

    # Grow mask: slightly more permissive for connectivity
    candidate_grow = candidate_core.copy()

    if dilate_radius > 0:
        candidate_grow = binary_dilation(candidate_grow, ball(dilate_radius))
        candidate_grow = ndi.binary_closing(candidate_grow, structure=ball(1))
        candidate_grow = candidate_grow & artery_territory

        # Apply the bowel-boundary filter again after dilation,
        # because dilation can reconnect bowel-wall edges.
        candidate_grow = remove_bowel_boundary_like_components(
            mask=candidate_grow,
            ct_crop=ct_crop,
            protected=sma_valid,
            air_hu=air_hu,
            air_radius=air_radius,
            min_area=20,
            min_long_axis=12,
            min_aspect=2.5,
            axis=2,
        )

        candidate_grow = candidate_grow | sma_valid

    return (
        candidate_core.astype(bool),
        candidate_grow.astype(bool),
        artery_territory.astype(bool),
    )

def competitive_grow_gpu(candidate, sma_seed, vein_seed, max_iter=700, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cand = torch.from_numpy(candidate.astype(np.bool_)).to(device)
    sma = torch.from_numpy(sma_seed.astype(np.bool_)).to(device) & cand
    vein = torch.from_numpy(vein_seed.astype(np.bool_)).to(device) & cand
    cand = cand | sma | vein

    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    # 18-neighborhood: corner 제외, center 제외
    for i in [0, 2]:
        for j in [0, 2]:
            for k in [0, 2]:
                kernel[0, 0, i, j, k] = 0
    kernel[0, 0, 1, 1, 1] = 0

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

        conflict = sma_new & vein_new
        sma_new = sma_new & (~conflict)
        vein_new = vein_new & (~conflict)

        n_new = int(sma_new.sum().item() + vein_new.sum().item())
        if n_new == 0:
            print(f"\nConverged at iteration {i}")
            break

        sma_cur |= sma_new
        vein_cur |= vein_new
        sma_front = sma_new
        vein_front = vein_new

        if i % 20 == 0:
            print(f"\rIter {i} | SMA: {int(sma_cur.sum())} | SMV: {int(vein_cur.sum())}", end="", flush=True)

    print(flush=True)
    return (
        sma_cur.squeeze().detach().cpu().numpy().astype(bool),
        vein_cur.squeeze().detach().cpu().numpy().astype(bool),
    )


def filter_by_sma_cross_section_area(vessel, sma, axis=2, scale=1.8):
    sma_areas = []
    for z in range(sma.shape[axis]):
        sma_slice = np.take(sma, z, axis=axis)
        sma_areas.append(sma_slice.sum())

    max_allowed_area = max(sma_areas) * scale
    filtered = np.zeros_like(vessel, dtype=bool)
    structure2d = ndi.generate_binary_structure(2, 2)

    for z in range(vessel.shape[axis]):
        vessel_slice = np.take(vessel, z, axis=axis)
        labeled, n = ndi.label(vessel_slice, structure=structure2d)
        kept_slice = np.zeros_like(vessel_slice, dtype=bool)
        for lab in range(1, n + 1):
            comp = labeled == lab
            if comp.sum() <= max_allowed_area:
                kept_slice |= comp
        if axis == 0:
            filtered[z, :, :] = kept_slice
        elif axis == 1:
            filtered[:, z, :] = kept_slice
        else:
            filtered[:, :, z] = kept_slice
    return filtered


def connect_nearby_branches_by_path(
    artery_tree,
    candidate,
    ct_crop,
    max_dist=10,
    hu_min=90,
    hu_max=400,
    dilate_radius=0,
    max_components=10,
    max_comp_size=500,
):
    passable = candidate | ((ct_crop >= hu_min) & (ct_crop <= hu_max))
    passable = passable & (ct_crop >= 0)

    labeled, _ = ndi.label(candidate)
    main = artery_tree.astype(bool)
    main_coords = np.argwhere(main)
    if len(main_coords) == 0:
        return artery_tree

    tree = cKDTree(main_coords)
    out = main.copy()
    objs = ndi.find_objects(labeled)
    processed = 0

    for lab, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        comp = labeled[slc] == lab
        comp_size = int(comp.sum())
        if comp_size < 3 or comp_size > max_comp_size:
            continue
        comp_global = np.argwhere(comp) + np.array([s.start for s in slc])
        dists, idxs = tree.query(comp_global, k=1)
        min_i = np.argmin(dists)
        if dists[min_i] > max_dist:
            continue

        start = tuple(main_coords[idxs[min_i]])
        end = tuple(comp_global[min_i])
        lo = np.maximum(np.minimum(start, end) - max_dist - 5, 0)
        hi = np.minimum(np.maximum(start, end) + max_dist + 6, candidate.shape)
        local_slices = tuple(slice(lo[d], hi[d]) for d in range(3))
        local_passable = passable[local_slices]
        local_start = tuple(np.array(start) - lo)
        local_end = tuple(np.array(end) - lo)
        cost = np.where(local_passable, 1.0, 1e5)

        try:
            path, cost_val = route_through_array(cost, local_start, local_end, fully_connected=True)
        except Exception:
            continue
        if cost_val > max_dist * 4:
            continue

        path = np.array(path) + lo
        out[tuple(path.T)] = True
        out[tuple(comp_global.T)] = True
        processed += 1
        if processed >= max_components:
            break

    if dilate_radius > 0:
        out = binary_dilation(out, ball(dilate_radius))
        out = out & passable

    print("Connected components:", processed)
    return out
def add_tube_path(out, path, radius=2):
    tube = np.zeros_like(out, dtype=bool)
    path = np.asarray(path)
    tube[tuple(path.T)] = True
    tube = binary_dilation(tube, ball(radius))
    return out | tube


def connect_valid_vessel_gaps(
    artery_tree,
    candidate,
    ct_crop,
    max_dist=8,
    hu_min=-30,
    hu_max=400,
    min_align=0.5,
):
    skel = skeletonize_3d(artery_tree).astype(bool)

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    neighbor_count = ndi.convolve(
        skel.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

    endpoints = skel & (neighbor_count == 1)
    ep_coords = np.argwhere(endpoints)

    if len(ep_coords) < 2:
        print("Valid vessel gap connections: 0")
        return artery_tree

    passable = candidate | ((ct_crop >= hu_min) & (ct_crop <= hu_max))
    passable = passable & (ct_crop >= hu_min)

    def endpoint_dir(p, r=5):
        p = np.asarray(p)
        lo = np.maximum(p - r, 0)
        hi = np.minimum(p + r + 1, skel.shape)
        slc = tuple(slice(lo[d], hi[d]) for d in range(3))

        pts = np.argwhere(skel[slc]) + lo
        if len(pts) < 3:
            return None

        pts0 = pts - p
        cov = pts0.T @ pts0
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]

        mean_vec = pts0.mean(axis=0)
        if np.dot(v, mean_vec) < 0:
            v = -v

        return v / (np.linalg.norm(v) + 1e-6)

    dirs = [endpoint_dir(p) for p in ep_coords]

    tree = cKDTree(ep_coords)
    pairs = list(tree.query_pairs(r=max_dist))
    pairs.sort(key=lambda ij: np.linalg.norm(ep_coords[ij[0]] - ep_coords[ij[1]]))

    out = artery_tree.copy()
    connected = 0

    for i, j in pairs:
        p1 = ep_coords[i]
        p2 = ep_coords[j]

        v1 = dirs[i]
        v2 = dirs[j]

        if v1 is None or v2 is None:
            continue

        gap = p2 - p1
        dist = np.linalg.norm(gap)

        if dist < 2 or dist > max_dist:
            continue

        gap_dir = gap / dist

        if np.dot(v1, gap_dir) < min_align:
            continue
        if np.dot(v2, -gap_dir) < min_align:
            continue

        lo = np.maximum(np.minimum(p1, p2) - 5, 0)
        hi = np.minimum(np.maximum(p1, p2) + 6, artery_tree.shape)

        slc = tuple(slice(lo[d], hi[d]) for d in range(3))
        local_passable = passable[slc]

        start = tuple(p1 - lo)
        end = tuple(p2 - lo)

        cost = np.where(local_passable, 1.0, 1e6)

        try:
            path, cost_val = route_through_array(
                cost,
                start,
                end,
                fully_connected=True,
            )
        except Exception:
            continue

        if cost_val > max_dist * 2.5:
            continue

        path = np.array(path) + lo
        out = add_tube_path(out, path, radius=2)
        connected += 1

    print("Valid vessel gap connections:", connected)
    return out | artery_tree
def remove_bowel_boundary_like_components(
    mask,
    ct_crop,
    protected=None,
    air_hu=-500,
    air_radius=2,
    min_area=20,
    min_long_axis=12,
    min_aspect=2.5,
    axis=2,
):
    """
    Remove long/linear components near air-filled bowel lumen.
    This does NOT remove all vessels near bowel.
    It removes only air-adjacent components that look like long bowel-wall boundaries in 2D slices.
    """
    air = ct_crop <= air_hu
    near_air = binary_dilation(air, ball(air_radius))

    suspect = mask & near_air

    if protected is None:
        protected = np.zeros_like(mask, dtype=bool)

    remove = np.zeros_like(mask, dtype=bool)
    structure2d = ndi.generate_binary_structure(2, 2)

    for z in range(mask.shape[axis]):
        s = np.take(suspect, z, axis=axis)
        p = np.take(protected, z, axis=axis)

        labeled, n = ndi.label(s, structure=structure2d)
        remove_slice = np.zeros_like(s, dtype=bool)

        for lab in range(1, n + 1):
            comp = labeled == lab
            area = int(comp.sum())
            if area < min_area:
                continue

            coords = np.argwhere(comp)
            if len(coords) == 0:
                continue

            span = coords.max(axis=0) - coords.min(axis=0) + 1
            long_axis = int(span.max())
            short_axis = int(max(span.min(), 1))
            aspect = long_axis / short_axis

            # bowel boundary: long, thin/arc-like component near air
            if long_axis >= min_long_axis and aspect >= min_aspect:
                remove_slice |= comp

        # never remove protected seed voxels
        remove_slice = remove_slice & (~p)

        if axis == 0:
            remove[z, :, :] = remove_slice
        elif axis == 1:
            remove[:, z, :] = remove_slice
        else:
            remove[:, :, z] = remove_slice

    print("Bowel-boundary-like voxels removed:", int(remove.sum()))

    out = mask & (~remove)
    out = out | protected
    return out

def main():
    start_time = time.time()
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", default="/data/drdcad/datasets/private/SmallBowelObstruction_7Apr2026/Data/anon_niftis/00004/00001/00004_00001_3.nii.gz")
    parser.add_argument("--seg", default="00004_00001_3_sma_smv.nii.gz")
    parser.add_argument("--out", default="00004_00001_3_gt_algo.nii.gz")
    parser.add_argument("--hu_min", type=float, default=95)
    parser.add_argument("--hu_max", type=float, default=370)
    parser.add_argument("--vesselness_thr", type=float, default=0.0012)
    parser.add_argument("--margin", type=int, default=95)
    parser.add_argument("--max_iter", type=int, default=700)
    parser.add_argument("--min_size", type=int, default=25)
    parser.add_argument("--vein_block_radius", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--connect_branches", type=int, default=0)
    parser.add_argument("--air_hu", type=float, default=-500)
    parser.add_argument("--air_radius", type=int, default=2)
    args = parser.parse_args()

    ct, affine, header = load_nifti(args.ct)

    print("shape:", ct.shape)
    print("affine:")
    print(affine)

    sma_smv, _, _ = load_nifti(args.seg)

    sma = sma_smv == 1
    vein = (sma_smv == 2) | (sma_smv == 3)

    roi_mask = sma | vein
    slices = crop_bbox(roi_mask, margin=args.margin)
    ct_crop = ct[slices]
    sma_crop = sma[slices]
    vein_crop = vein[slices]


    sma_valid = sma_crop & (ct_crop >= 0)
    vein_valid = vein_crop & (ct_crop >= 0)

    print("Artery seed voxels:", int(sma_valid.sum()))
    print("Vein seed voxels:", int(vein_valid.sum()))

    candidate_core, candidate, artery_territory = make_vessel_candidate(
        ct_crop=ct_crop,
        sma_valid=sma_valid,
        vein_valid=vein_valid,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        vesselness_thr=args.vesselness_thr,
        use_frangi=True,
        territory_ratio=2.0,
        dilate_radius=1,
        air_hu=args.air_hu,
        air_radius=args.air_radius,
    )

    print("Candidate core voxels:", int(candidate_core.sum()))
    print("Candidate grow voxels:", int(candidate.sum()))

    vein_exclude = binary_dilation(vein_valid, ball(args.vein_block_radius))
    vein_exclude = vein_exclude & (~sma_valid)
    print("Vein-exclude voxels:", int(vein_exclude.sum()))

    candidate = candidate & (~vein_exclude)
    candidate = candidate | sma_valid

    candidate_core = candidate_core & (~vein_exclude)
    candidate_core = candidate_core | sma_valid

    print("Growing SMA arterial tree with competitive vein growth...")
    artery_tree, vein_tree = competitive_grow_gpu(
        candidate=candidate,
        sma_seed=sma_valid,
        vein_seed=vein_valid,
        max_iter=args.max_iter,
        device=args.device,
    )

    print("Raw artery tree voxels:", int(artery_tree.sum()))
    print("Raw vein tree voxels:", int(vein_tree.sum()))

    artery_tree = filter_by_sma_cross_section_area(
        vessel=artery_tree,
        sma=sma_valid,
        axis=2,
        scale=1.8,
    )
    print("After area filter voxels:", int(artery_tree.sum()))

    artery_tree = keep_connected_to_seed(candidate=artery_tree | sma_valid, seed=sma_valid)
    print("After keep connected voxels:", int(artery_tree.sum()))

    artery_tree = artery_tree & (~vein_tree)
    artery_tree = remove_small_objects(artery_tree.astype(bool), min_size=args.min_size)
    artery_tree = artery_tree | sma_valid
    print("After vein/small-object removal voxels:", int(artery_tree.sum()))

    # ============================================================
    # Strict cleanup only
    # No skeleton refinement, no endpoint gap fill, no branch connection
    # ============================================================

    strict_mask = candidate_core & artery_territory
    strict_mask = strict_mask & (~vein_exclude)
    strict_mask = strict_mask | sma_valid

    # Do not add new voxels.
    # Only keep the already-grown artery inside the strict candidate mask.
    artery_tree = artery_tree & strict_mask
    artery_tree = artery_tree | sma_valid

    artery_tree = remove_small_objects(
        artery_tree.astype(bool),
        min_size=args.min_size
    )
    artery_tree = remove_bowel_boundary_like_components(
        mask=artery_tree,
        ct_crop=ct_crop,
        protected=sma_valid,
        air_hu=args.air_hu,
        air_radius=args.air_radius,
        min_area=20,
        min_long_axis=12,
        min_aspect=2.5,
        axis=2,
    )

    artery_tree = artery_tree | sma_valid
    
    print("After strict cleanup voxels:", int(artery_tree.sum()))
    artery_tree = keep_connected_to_seed(candidate=artery_tree | sma_valid, seed=sma_valid)
    
    structure6 = ndi.generate_binary_structure(3, 1)
    labeled, n = ndi.label(artery_tree, structure=structure6)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    main_label = sizes.argmax()
    artery_tree = labeled == main_label

    # 1) 먼저 SMA seed 포함
    artery_tree = artery_tree | sma_valid

    # 2) 그다음 largest component만 유지
    structure6 = ndi.generate_binary_structure(3, 1)
    labeled, n = ndi.label(artery_tree, structure=structure6)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    main_label = sizes.argmax()
    artery_tree = labeled == main_label

    # 3) 여기서는 다시 artery_tree | sma_valid 하지 말기
    _, n6 = ndi.label(artery_tree, structure=structure6)
    print("Final connected components 6-connectivity:", n6)

    result = np.zeros(sma.shape, dtype=np.uint8)
    result[slices] = artery_tree.astype(np.uint8)
    save_nifti(result, affine, header, args.out)

    print("Saved:", args.out)
    print("Output voxels:", int(result.sum()))
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Total runtime: {minutes} min {seconds:.2f} sec")
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
