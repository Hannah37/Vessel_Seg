import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.utils import set_determinism

from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler


import random
import numpy as np
import blosc2
import torch
from torch.utils.data import Dataset as TorchDataset


def strip_nii_suffix(name):
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def case_id_from_item(item):
    """Extract nnU-Net case id from a split entry.

    Works with JSON entries such as:
      {"image": "/path/CASE_0000.nii.gz", "label": "/path/CASE.nii.gz"}
    and returns CASE.
    """
    if isinstance(item, dict):
        if "case_id" in item:
            return str(item["case_id"])
        path = item.get("image", None) or item.get("label", None)
    else:
        path = item

    name = Path(str(path)).name
    case_id = strip_nii_suffix(name)

    # nnU-Net raw image channel suffix, e.g. CASE_0000.nii.gz -> CASE
    if len(case_id) > 5 and case_id[-5] == "_" and case_id[-4:].isdigit():
        case_id = case_id[:-5]

    # Safety: labels may have accidental suffixes in custom folders.
    for suffix in ["_gt", "_sma_smv", "_seg"]:
        if case_id.endswith(suffix):
            case_id = case_id[: -len(suffix)]

    return case_id


def resolve_case_file(preprocessed_dir, case_id):
    """Return the nnU-Net preprocessed image file for a case.

    Your folder is nnU-Net v2's default layout:
      CASE.b2nd
      CASE_seg.b2nd
      CASE.pkl

    This also supports older/unpacked .npy and .npz layouts as fallback.
    """
    preprocessed_dir = Path(preprocessed_dir)
    stems = [case_id, case_id.replace("_gt", ""), case_id.replace("_sma_smv", "")]

    # Prefer the current nnU-Net v2 b2nd image file. Do not select *_seg.b2nd.
    for stem in stems:
        p = preprocessed_dir / f"{stem}.b2nd"
        if p.exists():
            return p

    # Fallbacks for older nnU-Net/preprocessed layouts.
    for stem in stems:
        for suffix in [".npy", ".npz"]:
            p = preprocessed_dir / f"{stem}{suffix}"
            if p.exists():
                return p

    # Broad search, excluding segmentation sidecars.
    patterns = []
    for suffix in [".b2nd", ".npy", ".npz"]:
        patterns.extend(sorted(preprocessed_dir.glob(f"{case_id}*{suffix}")))
    matches = [p for p in patterns if not p.stem.endswith("_seg") and "_seg" not in p.stem]

    if len(matches) == 1:
        return matches[0]

    example = ", ".join([p.name for p in sorted(preprocessed_dir.glob("*"))[:12]])
    raise FileNotFoundError(
        f"Cannot find unique preprocessed image file for case_id='{case_id}' in {preprocessed_dir}. "
        f"Expected files like {case_id}.b2nd and {case_id}_seg.b2nd. "
        f"Examples in folder: {example}"
    )


def _load_b2nd_array(path):
    """Read a nnU-Net v2 .b2nd file into a NumPy array."""
    arr = blosc2.open(urlpath=str(path), mode="r")
    return arr[:]


def _load_from_b2nd(image_path):
    image_path = Path(image_path)
    label_path = image_path.with_name(image_path.stem + "_seg.b2nd")

    if not label_path.exists():
        raise FileNotFoundError(
            f"Cannot find segmentation sidecar for {image_path}. Expected {label_path.name}."
        )

    image = _load_b2nd_array(image_path)
    label = _load_b2nd_array(label_path)
    return image, label


def _load_from_npz(npz_path):
    with np.load(npz_path) as f:
        keys = list(f.keys())
        if "data" in keys and "seg" in keys:
            image = f["data"]
            label = f["seg"]
        elif "data" in keys:
            arr = f["data"]
            if arr.ndim == 4 and arr.shape[0] >= 2:
                image = arr[:-1]
                label = arr[-1:]
            else:
                raise ValueError(f"{npz_path} has key 'data' only with unsupported shape {arr.shape}")
        elif "image" in keys and "label" in keys:
            image = f["image"]
            label = f["label"]
        else:
            raise ValueError(f"Unsupported keys in {npz_path}: {keys}")
    return image, label


def _load_from_npy(npy_path):
    arr = np.load(npy_path, mmap_mode="r")
    base = Path(npy_path)

    seg_candidates = [
        base.with_name(base.stem + "_seg.npy"),
        base.with_name(base.stem + ".seg.npy"),
    ]
    seg_file = next((p for p in seg_candidates if p.exists()), None)

    if seg_file is not None:
        image = arr
        label = np.load(seg_file, mmap_mode="r")
    else:
        if arr.ndim == 4 and arr.shape[0] >= 2:
            image = arr[:-1]
            label = arr[-1:]
        else:
            npz = base.with_suffix(".npz")
            if npz.exists():
                return _load_from_npz(npz)
            raise ValueError(
                f"Could not find segmentation for {npy_path}. Expected *_seg.npy or packed .npz."
            )
    return image, label


def load_nnunet_case(case_file):
    """Load one preprocessed nnU-Net case as image [1,D,H,W], label [1,D,H,W]."""
    case_file = Path(case_file)

    if case_file.suffix == ".b2nd":
        image, label = _load_from_b2nd(case_file)
    elif case_file.suffix == ".npz":
        npy = case_file.with_suffix(".npy")
        if npy.exists():
            image, label = _load_from_npy(npy)
        else:
            image, label = _load_from_npz(case_file)
    elif case_file.suffix == ".npy":
        image, label = _load_from_npy(case_file)
    else:
        raise ValueError(f"Unsupported preprocessed file: {case_file}")

    if image.ndim == 3:
        image = image[None]
    if label.ndim == 3:
        label = label[None]

    # Use the first image channel and binary foreground label for vessel segmentation.
    image = np.asarray(image[:1], dtype=np.float32)
    label = (np.asarray(label[:1]) > 0).astype(np.uint8, copy=False)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image, label


def pad_to_roi(image, label, roi_size):
    spatial = image.shape[1:]
    pad_width = [(0, 0)]
    for s, r in zip(spatial, roi_size):
        total = max(0, r - s)
        before = total // 2
        after = total - before
        pad_width.append((before, after))

    if any(p != (0, 0) for p in pad_width[1:]):
        image = np.pad(image, pad_width, mode="constant", constant_values=0)
        label = np.pad(label, pad_width, mode="constant", constant_values=0)
    return image, label


def crop_around_center(image, label, center, roi_size):
    image, label = pad_to_roi(image, label, roi_size)
    spatial = image.shape[1:]
    starts = []
    for c, s, r in zip(center, spatial, roi_size):
        start = int(c) - r // 2
        start = max(0, min(start, s - r))
        starts.append(start)
    z, y, x = starts
    dz, dy, dx = roi_size
    slices = (slice(None), slice(z, z + dz), slice(y, y + dy), slice(x, x + dx))
    return image[slices], label[slices]


def random_patch(image, label, roi_size, pos_ratio=2.0 / 3.0):
    spatial = image.shape[1:]
    use_pos = random.random() < pos_ratio
    fg = np.argwhere(label[0] > 0)
    if use_pos and len(fg) > 0:
        center = fg[random.randrange(len(fg))]
    else:
        center = np.array([random.randrange(max(1, s)) for s in spatial], dtype=np.int64)
    return crop_around_center(image, label, center, roi_size)


def augment_patch(image, label):
    for spatial_axis in range(3):
        if random.random() < 0.10:
            axis = spatial_axis + 1
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    if random.random() < 0.10:
        k = random.randint(1, 3)
        axes = random.choice([(1, 2), (1, 3), (2, 3)])
        image = np.rot90(image, k=k, axes=axes)
        label = np.rot90(label, k=k, axes=axes)
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


class NnUNetPreprocessedDataset(TorchDataset):
    """Read nnU-Net v2 preprocessed files directly.

    mode='full': whole preprocessed volume.
    mode='patch': foreground-biased patches, num_samples per case.
    """

    def __init__(self, items, preprocessed_dir, roi_size=(128, 128, 128), mode="patch", num_samples=2, augment=False):
        self.items = list(items)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.roi_size = tuple(int(v) for v in roi_size)
        self.mode = mode
        self.num_samples = int(num_samples)
        self.augment = bool(augment)
        self.case_ids = [case_id_from_item(it) for it in self.items]
        self.case_files = [resolve_case_file(self.preprocessed_dir, cid) for cid in self.case_ids]

    def __len__(self):
        return len(self.case_files)

    def __getitem__(self, idx):
        image, label = load_nnunet_case(self.case_files[idx])
        case_id = self.case_ids[idx]
        if self.mode == "full":
            return {
                "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
                "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
                "case_id": case_id,
            }
        images = []
        labels = []
        for _ in range(self.num_samples):
            img_patch, lab_patch = random_patch(image, label, self.roi_size)
            if self.augment:
                img_patch, lab_patch = augment_patch(img_patch, lab_patch)
            images.append(torch.from_numpy(img_patch).float())
            labels.append(torch.from_numpy(lab_patch).long())
        return {
            "image": torch.stack(images, dim=0),
            "label": torch.stack(labels, dim=0),
            "case_id": case_id,
        }


def flatten_patch_batch(batch_data, device):
    inputs = batch_data["image"]
    labels = batch_data["label"]
    if inputs.ndim == 6:
        b, n = inputs.shape[:2]
        inputs = inputs.reshape(b * n, *inputs.shape[2:])
        labels = labels.reshape(b * n, *labels.shape[2:])
    return inputs.to(device), labels.to(device).long()

def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_fold_json(json_path, fold):
    with open(json_path, "r") as f:
        data_json = json.load(f)
    if "folds" in data_json:
        fold_data = data_json["folds"][str(fold)]
        return fold_data["training"], fold_data["validation"]
    return data_json["training"], data_json["validation"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--preprocessed_dir", default="/data/drdcad/datasets_nnUNet/nnUNet_preprocessed/Dataset101_SBOvessel/nnUNetPlans_3d_fullres")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--full_val_every", type=int, default=200)
    parser.add_argument("--quick_val_max_cases", type=int, default=5)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_patch_samples", type=int, default=2)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class DiffUNetBinary(nn.Module):
    def __init__(self, diffusion_steps=1000, sample_steps=10, noise_schedule="linear", base_channels=32):
        super().__init__()
        f = base_channels
        features = [f, f, f * 2, f * 4, f * 8, f]
        self.num_label_channels = 1
        self.diffusion_steps = diffusion_steps
        self.sample_steps = sample_steps
        self.embed_model = BasicUNetEncoder(spatial_dims=3, in_channels=1, out_channels=2, features=features)
        self.model = BasicUNetDe(spatial_dims=3, in_channels=2, out_channels=1, features=features, act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(diffusion_steps, [diffusion_steps]), betas=betas, model_mean_type=ModelMeanType.START_X, model_var_type=ModelVarType.FIXED_LARGE, loss_type=LossType.MSE)
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(diffusion_steps, [sample_steps]), betas=betas, model_mean_type=ModelMeanType.START_X, model_var_type=ModelVarType.FIXED_LARGE, loss_type=LossType.MSE)
        self.sampler = UniformSampler(diffusion_steps)

    def q_sample(self, x_start):
        noise = torch.randn_like(x_start).to(x_start.device)
        t, _ = self.sampler.sample(x_start.shape[0], x_start.device)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)
        return x_t, t, noise

    def denoise(self, image, x_t, step):
        embeddings = self.embed_model(image)
        return self.model(x_t, t=step, image=image, embeddings=embeddings)

    def ddim_sample(self, image, spatial_size):
        embeddings = self.embed_model(image)
        b = image.shape[0]
        shape = (b, self.num_label_channels, spatial_size[0], spatial_size[1], spatial_size[2])
        sample_out = self.sample_diffusion.ddim_sample_loop(self.model, shape, model_kwargs={"image": image, "embeddings": embeddings})
        return sample_out["pred_xstart"]


def binary_dice_score(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    intersection = torch.sum(pred * target, dim=dims)
    denominator = torch.sum(pred, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean().item()


def main():
    args = get_args()
    set_determinism(seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_files = load_fold_json(args.json, args.fold)
    if args.quick_val_max_cases > 0:
        quick_val_files = val_files[: args.quick_val_max_cases]
    else:
        quick_val_files = val_files
    roi_size = (args.roi_x, args.roi_y, args.roi_z)

    print(f"Fold: {args.fold}")
    print(f"Train cases: {len(train_files)}")
    print(f"Val cases: {len(val_files)}")
    print(f"Quick val cases: {len(quick_val_files)}")
    print(f"Preprocessed dir: {args.preprocessed_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Diffusion steps: {args.diffusion_steps} | Sample steps: {args.sample_steps} | Base channels: {args.base_channels}")

    train_ds = NnUNetPreprocessedDataset(train_files, args.preprocessed_dir, roi_size, mode="patch", num_samples=args.num_patch_samples, augment=True)
    val_patch_ds = NnUNetPreprocessedDataset(quick_val_files, args.preprocessed_dir, roi_size, mode="patch", num_samples=args.num_patch_samples, augment=False)
    val_full_ds = NnUNetPreprocessedDataset(val_files, args.preprocessed_dir, roi_size, mode="full", num_samples=1, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_patch_loader = DataLoader(val_patch_ds, batch_size=1, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)
    val_full_loader = DataLoader(val_full_ds, batch_size=1, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffUNetBinary(args.diffusion_steps, args.sample_steps, args.noise_schedule, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def poly_lr_lambda(epoch):
        return max(0.0, 1.0 - epoch / args.max_epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
    dice_loss = DiceLoss(sigmoid=True, include_background=True)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_pseudo_metric = -1.0
    best_pseudo_epoch = -1
    best_full_metric = -1.0
    best_full_epoch = -1
    latest_ckpt = out_dir / "latest_model.pt"
    best_quick_ckpt = out_dir / "best_model.pt"
    best_full_ckpt = out_dir / "best_full_model.pt"
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    def save_ckpt(path, epoch, extra=None):
        payload = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_pseudo_metric": best_pseudo_metric, "best_pseudo_epoch": best_pseudo_epoch, "best_full_metric": best_full_metric, "best_full_epoch": best_full_epoch, "args": vars(args)}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def run_quick_validation(epoch):
        nonlocal best_pseudo_metric, best_pseudo_epoch
        start = time.time()
        model.eval()
        dices = []
        with torch.no_grad():
            for val_data in val_patch_loader:
                val_inputs, val_labels = flatten_patch_batch(val_data, device)
                val_labels_f = (val_labels > 0).float()
                with torch.cuda.amp.autocast(enabled=args.amp):
                    val_outputs = model.ddim_sample(image=val_inputs, spatial_size=tuple(val_inputs.shape[2:]))
                val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
                dices.append(binary_dice_score(val_preds, val_labels_f))
        metric = float(sum(dices) / max(len(dices), 1))
        print(f"Quick validation epoch={epoch} | pseudo_dice={metric:.6f} | val_time={format_seconds(time.time()-start)}", flush=True)
        if metric > best_pseudo_metric:
            best_pseudo_metric = metric
            best_pseudo_epoch = epoch
            save_ckpt(best_quick_ckpt, epoch, {"selection_metric": "quick_pseudo_dice"})
            print(f"New best model saved by quick validation | pseudo_dice={metric:.6f} | epoch={epoch}", flush=True)
        return metric

    def run_full_validation(epoch, tag="Full validation"):
        nonlocal best_full_metric, best_full_epoch
        start = time.time()
        model.eval()
        dices = []
        with torch.no_grad():
            for val_data in val_full_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = (val_data["label"].to(device) > 0).float()
                def predictor(x):
                    return model.ddim_sample(image=x, spatial_size=tuple(x.shape[2:]))
                with torch.cuda.amp.autocast(enabled=args.amp):
                    val_outputs = sliding_window_inference(val_inputs, roi_size=roi_size, sw_batch_size=args.sw_batch_size, predictor=predictor, overlap=0.25)
                val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
                dices.append(binary_dice_score(val_preds, val_labels))
        metric = float(sum(dices) / max(len(dices), 1))
        elapsed = time.time() - start
        print(f"{tag} epoch={epoch} | mean_dice={metric:.6f} | val_time={format_seconds(elapsed)}", flush=True)
        if metric > best_full_metric:
            best_full_metric = metric
            best_full_epoch = epoch
            save_ckpt(best_full_ckpt, epoch, {"selection_metric": "full_validation_dice"})
            print(f"New best full-validation model saved | dice={metric:.6f} | epoch={epoch}", flush=True)
        return metric, elapsed

    total_start_time = time.time()
    print(f"Total epochs: {args.max_epochs}", flush=True)
    print(f"Steps per epoch: {len(train_loader)}", flush=True)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = epoch_dice = epoch_bce = epoch_mse = 0.0
        step = 0
        epoch_start = time.time()
        for batch_data in train_loader:
            step += 1
            images, labels = flatten_patch_batch(batch_data, device)
            labels = (labels > 0).float()
            x_start = labels * 2.0 - 1.0
            x_t, t, _ = model.q_sample(x_start)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_xstart = model.denoise(image=images, x_t=x_t, step=t)
                loss_dice = dice_loss(pred_xstart, labels)
                loss_bce = bce_loss(pred_xstart, labels)
                loss_mse = mse_loss(torch.sigmoid(pred_xstart), labels)
                loss = loss_dice + loss_bce + loss_mse
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item(); epoch_dice += loss_dice.item(); epoch_bce += loss_bce.item(); epoch_mse += loss_mse.item()

        epoch_loss /= max(step, 1); epoch_dice /= max(step, 1); epoch_bce /= max(step, 1); epoch_mse /= max(step, 1)
        scheduler.step()
        elapsed_total = time.time() - total_start_time
        eta = (args.max_epochs - epoch) * (elapsed_total / max(epoch, 1))
        finish = datetime.now() + timedelta(seconds=eta)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.max_epochs} finished | loss={epoch_loss:.6f} | dice_loss={epoch_dice:.6f} | bce={epoch_bce:.6f} | mse={epoch_mse:.6f} | lr={current_lr:.6g} | epoch_time={format_seconds(time.time()-epoch_start)} | elapsed={format_seconds(elapsed_total)} | ETA={format_seconds(eta)} | estimated_finish={finish.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        save_ckpt(latest_ckpt, epoch)
        if args.val_every > 0 and epoch % args.val_every == 0:
            run_quick_validation(epoch)
        if args.full_val_every > 0 and epoch % args.full_val_every == 0:
            run_full_validation(epoch, tag="Full validation")

    final_metric, final_val_time = run_full_validation(args.max_epochs, tag="Final full validation")
    metrics = {"fold": args.fold, "max_epochs": args.max_epochs, "best_pseudo_dice": float(best_pseudo_metric), "best_pseudo_epoch": int(best_pseudo_epoch), "best_full_dice": float(best_full_metric), "best_full_epoch": int(best_full_epoch), "final_dice": float(final_metric), "final_epoch": int(args.max_epochs), "final_val_time_sec": float(final_val_time), "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("==========================================", flush=True)
    print("Training finished", flush=True)
    print(f"Fold: {args.fold}", flush=True)
    print(f"Best pseudo Dice: {best_pseudo_metric:.6f} at epoch {best_pseudo_epoch}", flush=True)
    print(f"Best full Dice: {best_full_metric:.6f} at epoch {best_full_epoch}", flush=True)
    print(f"Final Dice: {final_metric:.6f} at epoch {args.max_epochs}", flush=True)
    print(f"Final metrics saved: {out_dir / 'final_metrics.json'}", flush=True)
    print("==========================================", flush=True)


if __name__ == "__main__":
    main()
