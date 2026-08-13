#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from clm_net.model.CLM_Net import CLM_Net
except ModuleNotFoundError:
    from clm_net.model.CLM_Net import CLM_Net


class VarroaBBoxMaskDataset(Dataset):
    """Varroa dataset with rectangular segmentation masks generated from bbox labels."""

    def __init__(self, root=".", split="train", transform=None, size_divisor=16):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.size_divisor = size_divisor
        split_root = self._resolve_split_root(self.root, split)
        self.image_dir = split_root / "videos"
        self.label_dir = split_root / "labels"
        self.images = sorted(self.image_dir.rglob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No PNG images found under {self.image_dir}")

    @staticmethod
    def _resolve_split_root(root, split):
        candidates = [
            root / split,
            root / split / split,
            root,
        ]
        for candidate in candidates:
            if (candidate / "videos").is_dir() and (candidate / "labels").is_dir():
                return candidate

        checked = "\n".join(str(candidate / "videos") for candidate in candidates)
        raise FileNotFoundError(
            f"Could not find videos/labels for split '{split}'. Checked:\n{checked}"
        )

    def __len__(self):
        return len(self.images)

    def _label_path(self, image_path):
        rel = image_path.relative_to(self.image_dir)
        return (self.label_dir / rel).with_suffix(".txt")

    @staticmethod
    def _read_boxes(label_path):
        if not label_path.exists():
            return []

        lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
        boxes = []
        for line in lines[1:]:
            values = [float(x) for x in line.replace(",", " ").split()]
            for i in range(0, len(values) - 3, 4):
                boxes.append(values[i:i + 4])
        return boxes

    @staticmethod
    def _boxes_to_mask(boxes, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            left = int(np.floor(min(x1, x2)))
            right = int(np.ceil(max(x1, x2)))
            top = int(np.floor(min(y1, y2)))
            bottom = int(np.ceil(max(y1, y2)))

            left = max(0, min(left, width - 1))
            right = max(0, min(right, width))
            top = max(0, min(top, height - 1))
            bottom = max(0, min(bottom, height))

            if right > left and bottom > top:
                mask[top:bottom, left:right] = 1
        return mask

    def _pad_to_divisor(self, image, mask):
        if not self.size_divisor:
            return image, mask

        height, width = mask.shape[:2]
        padded_height = int(np.ceil(height / self.size_divisor) * self.size_divisor)
        padded_width = int(np.ceil(width / self.size_divisor) * self.size_divisor)
        pad_bottom = padded_height - height
        pad_right = padded_width - width

        if pad_bottom == 0 and pad_right == 0:
            return image, mask

        image = cv2.copyMakeBorder(
            image,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        mask = cv2.copyMakeBorder(
            mask,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        return image, mask

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        boxes = self._read_boxes(self._label_path(image_path))
        mask = self._boxes_to_mask(boxes, width=width, height=height)
        image, mask = self._pad_to_divisor(image, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()

        return image, mask


def get_transforms(input_height, input_width, train=True, resize_mode="pad"):
    if resize_mode == "stretch":
        transforms = [A.Resize(input_height, input_width)]
    elif resize_mode == "pad":
        transforms = [
            A.PadIfNeeded(
                min_height=input_height,
                min_width=input_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            )
        ]
    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")

    if train:
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.25),
        ]
    transforms += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def dice_loss(pred_probs, target_one_hot, smooth=1.0):
    intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
    total = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    return 1 - ((2.0 * intersection + smooth) / (total + smooth)).mean()


def iou_loss(pred_probs, target_one_hot, smooth=1.0):
    intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
    union = (pred_probs + target_one_hot - pred_probs * target_one_hot).sum(dim=(2, 3))
    return 1 - ((intersection + smooth) / (union + smooth)).mean()


def focal_loss(pred_logits, target_long, alpha=0.7, gamma=2.0):
    ce_loss = F.cross_entropy(pred_logits, target_long, reduction="none")
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()


class UltimateCombinedLoss(nn.Module):
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        iou_weight=1.0,
        focal_weight=1.0,
        boundary_weight=0.0,
        connectivity_weight=0.0,
        class_weights=(0.2, 0.8),
        device="cuda",
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.iou_weight = float(iou_weight)
        self.focal_weight = float(focal_weight)
        self.boundary_weight = float(boundary_weight)
        self.connectivity_weight = float(connectivity_weight)
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
        self.kernel = torch.ones(5, 5, device=device)

    @torch.no_grad()
    def _distance_transform_bg(self, target_positive):
        bg = target_positive.cpu().numpy() == 0
        return torch.from_numpy(distance_transform_edt(bg)).float().to(target_positive.device)

    def get_relaxed_boundary_loss(self, pred_probs, target_one_hot):
        from kornia.morphology import dilation

        pred_positive = pred_probs[:, 1:2]
        target_positive = target_one_hot[:, 1:2]
        boundary_zone = dilation(target_positive, self.kernel)
        target_dist_map = self._distance_transform_bg(target_positive)
        return (torch.abs(pred_positive - target_positive) * target_dist_map * boundary_zone).mean()

    def get_soft_connectivity_loss(self, pred_probs):
        from kornia.morphology import erosion

        pred_positive = pred_probs[:, 1:2]
        return (pred_positive - erosion(pred_positive, self.kernel)).mean()

    def forward(self, pred_logits, target, return_components=False):
        target_long = target.long()
        pred_probs = F.softmax(pred_logits, dim=1)
        target_one_hot = F.one_hot(target_long, num_classes=pred_logits.shape[1]).permute(0, 3, 1, 2).float()

        loss_ce = self.ce(pred_logits, target_long) if self.ce_weight else pred_logits.new_tensor(0.0)
        loss_dice = dice_loss(pred_probs, target_one_hot) if self.dice_weight else pred_logits.new_tensor(0.0)
        loss_iou = iou_loss(pred_probs, target_one_hot) if self.iou_weight else pred_logits.new_tensor(0.0)
        loss_focal = focal_loss(pred_logits, target_long) if self.focal_weight else pred_logits.new_tensor(0.0)
        loss_boundary = self.get_relaxed_boundary_loss(pred_probs, target_one_hot) if self.boundary_weight else pred_logits.new_tensor(0.0)
        loss_connectivity = self.get_soft_connectivity_loss(pred_probs) if self.connectivity_weight else pred_logits.new_tensor(0.0)

        total_loss = (
            self.ce_weight * loss_ce
            + self.dice_weight * loss_dice
            + self.iou_weight * loss_iou
            + self.focal_weight * loss_focal
            + self.boundary_weight * loss_boundary
            + self.connectivity_weight * loss_connectivity
        )

        if not return_components:
            return total_loss

        return total_loss, {
            "total_loss": float(total_loss.detach().item()),
            "ce_loss": float(loss_ce.detach().item()),
            "dice_loss": float(loss_dice.detach().item()),
            "iou_loss": float(loss_iou.detach().item()),
            "focal_loss": float(loss_focal.detach().item()),
            "boundary_loss": float(loss_boundary.detach().item()),
            "connectivity_loss": float(loss_connectivity.detach().item()),
        }


class GlobalSegmentationMetrics:
    def __init__(self, device, positive_class=1):
        self.device = device
        self.positive_class = positive_class
        self.reset()

    def reset(self):
        self.tp = torch.tensor(0.0, device=self.device)
        self.fp = torch.tensor(0.0, device=self.device)
        self.fn = torch.tensor(0.0, device=self.device)
        self.inter = torch.tensor(0.0, device=self.device)
        self.pred_sum = torch.tensor(0.0, device=self.device)
        self.target_sum = torch.tensor(0.0, device=self.device)
        self.union = torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def update(self, logits, targets):
        preds = torch.argmax(logits, dim=1)
        p = (preds == self.positive_class).float()
        t = (targets == self.positive_class).float()
        tp = (p * t).sum()
        self.tp += tp
        self.fp += (p * (1 - t)).sum()
        self.fn += ((1 - p) * t).sum()
        self.inter += tp
        self.pred_sum += p.sum()
        self.target_sum += t.sum()
        self.union += p.sum() + t.sum() - tp

    def compute(self):
        eps = 1e-6
        return {
            "dice": ((2 * self.inter + eps) / (self.pred_sum + self.target_sum + eps)).item(),
            "iou": ((self.inter + eps) / (self.union + eps)).item(),
            "precision": ((self.tp + eps) / (self.tp + self.fp + eps)).item(),
            "recall": ((self.tp + eps) / (self.tp + self.fn + eps)).item(),
        }


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path="best_lm_net_varroa.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_metrics = {}

    def __call__(self, val_loss, model, epoch, metrics):
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            self.best_metrics = metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "metrics": metrics,
                },
                self.path,
            )
            print(f"Saved checkpoint: {self.path}")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def run_epoch(model, loader, criterion, optimizer, device, train=True, max_grad_norm=1.0, scaler=None, amp=False):
    model.train(train)
    metric = GlobalSegmentationMetrics(device)
    total_loss = 0.0
    total_components = None
    skipped_nonfinite = 0
    desc = "Training" if train else "Validation"

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False, mininterval=0.5)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(images)
                loss, components = criterion(outputs, masks, return_components=True)

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                if train:
                    optimizer.zero_grad(set_to_none=True)
                tqdm.write(f"Warning: skipped non-finite loss batch ({desc}, skipped={skipped_nonfinite})")
                continue

            if train:
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            if total_components is None:
                total_components = {key: 0.0 for key in components}
            for key, value in components.items():
                total_components[key] += value

            total_loss += loss.item()
            metric.update(outputs.detach(), masks)
            current = metric.compute()
            pbar.set_postfix(dice=f"{current['dice']:.3f}", iou=f"{current['iou']:.3f}")

    num_batches = max(1, len(loader))
    effective_batches = max(1, num_batches - skipped_nonfinite)
    result = {"loss": total_loss / effective_batches, "skipped_nonfinite": skipped_nonfinite, **metric.compute()}
    if total_components:
        result.update({key: value / effective_batches for key, value in total_components.items()})
    return result


def append_metrics(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLM-Net on Varroa bbox-derived rectangular masks.")
    parser.add_argument("--root", default=".", help="Dataset root containing train/val/test folders.")
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--input-width", type=int, default=160)
    parser.add_argument("--size-divisor", type=int, default=16)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision training.")
    parser.add_argument(
        "--lnab-kind",
        "--skip-attention",
        dest="lnab_kind",
        choices=("partial", "identity"),
        default="partial",
        help="LNAB skip refinement. partial is the CLM-Net setting; identity disables LNAB.",
    )
    parser.add_argument(
        "--lnab-ratios",
        "--partial-ratios",
        dest="lnab_ratios",
        type=float,
        nargs=4,
        default=[0.5, 0.5, 0.5, 0.5],
        metavar=("LNAB1", "LNAB2", "LNAB3", "LNAB4"),
        help="Channel ratios attended by LNAB from low-res to high-res skips.",
    )
    parser.add_argument(
        "--lgb-bottleneck",
        "--gft-bottleneck",
        dest="lgb_bottleneck",
        type=int,
        default=192,
        help="Channels used inside the Lightweight Global Bottleneck.",
    )
    parser.add_argument(
        "--upsample-kind",
        choices=("psup", "bilinear_conv"),
        default="psup",
        help="Decoder upsampling block. bilinear_conv matches the original bilinear upsample + 3x3 conv.",
    )
    parser.add_argument(
        "--se-kind",
        choices=("sse", "se"),
        default="sse",
        help="Attention gate inside ReparamConv. sse is the CLM-Net lightweight default.",
    )
    parser.add_argument(
        "--filters",
        type=int,
        nargs=5,
        default=None,
        metavar=("F0", "F1", "F2", "F3", "F4"),
        help="Encoder/decoder channel list. Omit to keep the current default [24, 24, 48, 96, 192].",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("pad", "stretch"),
        default="pad",
        help="pad preserves image aspect ratio; stretch resizes to input-height/input-width.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint", default="best_clm_net_varroa.pt")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--connectivity-weight", type=float, default=0.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=1.0)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs/clm_net_varroa") / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    if args.checkpoint == "best_clm_net_varroa.pt":
        checkpoint_path = run_dir / "best.pt"
    log_dir = Path(args.log_dir)
    if args.log_dir == "logs":
        log_dir = run_dir

    train_dataset = VarroaBBoxMaskDataset(
        args.root,
        "train",
        get_transforms(args.input_height, args.input_width, train=True, resize_mode=args.resize_mode),
        size_divisor=args.size_divisor,
    )
    val_dataset = VarroaBBoxMaskDataset(
        args.root,
        "val",
        get_transforms(args.input_height, args.input_width, train=False, resize_mode=args.resize_mode),
        size_divisor=args.size_divisor,
    )
    test_dataset = VarroaBBoxMaskDataset(
        args.root,
        "test",
        get_transforms(args.input_height, args.input_width, train=False, resize_mode=args.resize_mode),
        size_divisor=args.size_divisor,
    )
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_filters = args.filters or [24, 24, 48, 96, 192]
    model = CLM_Net(
        channel=3,
        n_classes=2,
        filters=model_filters,
        lgb_bottleneck=args.lgb_bottleneck,
        lnab_kind=args.lnab_kind,
        lnab_ratios=args.lnab_ratios,
        upsample_kind=args.upsample_kind,
        se_kind=args.se_kind,
    ).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Resumed model weights from {args.resume}")

    criterion = UltimateCombinedLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        iou_weight=args.iou_weight,
        focal_weight=args.focal_weight,
        boundary_weight=args.boundary_weight,
        connectivity_weight=args.connectivity_weight,
        device=device,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=args.patience, path=str(checkpoint_path))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = vars(args).copy()
    config.update(
        {
            "model": "clm_net",
            "filters": model_filters,
            "lgb_bottleneck": args.lgb_bottleneck,
            "lnab_kind": args.lnab_kind,
            "lnab_ratios": args.lnab_ratios,
            "device": str(device),
            "checkpoint": str(checkpoint_path),
            "run_dir": str(run_dir),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "seed": args.seed,
                "root": str(Path(args.root).resolve()),
                "splits": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
                "model": "clm_net",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_log_path = log_dir / f"clm_net_varroa_{timestamp}.csv"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler, amp=use_amp)
        train_seconds = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False, amp=use_amp)
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start

        print(
            f"Train - Loss: {train_metrics['loss']:.4f} | Dice: {train_metrics['dice']:.4f} "
            f"| IoU: {train_metrics['iou']:.4f} | Precision: {train_metrics['precision']:.4f} "
            f"| Recall: {train_metrics['recall']:.4f}"
        )
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} "
            f"| IoU: {val_metrics['iou']:.4f} | Precision: {val_metrics['precision']:.4f} "
            f"| Recall: {val_metrics['recall']:.4f}"
        )
        print(
            f"Time  - Train: {train_seconds:.2f}s | Val: {val_seconds:.2f}s "
            f"| Epoch: {epoch_seconds:.2f}s"
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "epoch_seconds": epoch_seconds,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_metrics(csv_log_path, row)
        scheduler.step(val_metrics["loss"])
        early_stopping(val_metrics["loss"], model, epoch, val_metrics)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\nTraining completed.")
    print(f"CSV log saved at: {csv_log_path}")
    print(f"Best epoch: {early_stopping.best_epoch}")
    print(f"Best metrics: {early_stopping.best_metrics}")

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False, amp=use_amp)
    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False, amp=use_amp)
    evaluation = {
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "best_epoch": early_stopping.best_epoch,
        "val": val_metrics,
        "test": test_metrics,
    }
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Evaluation metrics saved at: {run_dir / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
