from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import LivenessImageDataset, LivenessSample, get_augmentation_config
from metrics import LivenessMetrics, binary_roc_auc, choose_security_threshold, select_threshold
from tinyliveness import build_liveness_model, count_parameters


MODEL_CHOICES = [
    "tiny",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_initial_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch_load(checkpoint_path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"init checkpoint missing keys: {len(missing)}")
    if unexpected:
        print(f"init checkpoint unexpected keys: {len(unexpected)}")
    print(f"initialized from checkpoint: {checkpoint_path}")


def make_balanced_sampler(dataset: LivenessImageDataset) -> WeightedRandomSampler:
    live_count = sum(sample.label == 1 for sample in dataset.samples)
    spoof_count = len(dataset.samples) - live_count
    class_weights = {
        1: 1.0 / max(live_count, 1),
        0: 1.0 / max(spoof_count, 1),
    }
    sample_weights = [
        class_weights[dataset.samples[index % len(dataset.samples)].label]
        for index in range(len(dataset))
    ]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


def make_loader(
    dataset: LivenessImageDataset,
    args: argparse.Namespace,
    device: torch.device,
    *,
    training: bool,
) -> DataLoader:
    sampler = make_balanced_sampler(dataset) if training and args.balanced_sampler else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=training and len(dataset) > args.batch_size,
    )


class BinaryLivenessLoss(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        live_weight: float,
        spoof_weight: float,
        focal_gamma: float,
        focal_pos_gamma: float,
        focal_neg_gamma: float,
        label_smoothing: float,
        dataset: LivenessImageDataset,
        balanced_sampler: bool,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.live_weight = float(live_weight)
        self.spoof_weight = float(spoof_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_pos_gamma = float(focal_pos_gamma)
        self.focal_neg_gamma = float(focal_neg_gamma)
        self.label_smoothing = float(label_smoothing)

        counts = dataset.class_counts()
        live = max(counts["live"], 1)
        spoof = max(counts["spoof"], 1)
        total = live + spoof
        if mode == "weighted_bce" and not balanced_sampler:
            self.live_weight *= total / (2.0 * live)
            self.spoof_weight *= total / (2.0 * spoof)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hard_labels = labels.float()
        if self.label_smoothing > 0:
            targets = hard_labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets = hard_labels

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = torch.where(
            hard_labels >= 0.5,
            torch.full_like(loss, self.live_weight),
            torch.full_like(loss, self.spoof_weight),
        )

        if self.mode in {"focal", "asymmetric_focal"}:
            probabilities = torch.sigmoid(logits)
            pt = torch.where(hard_labels >= 0.5, probabilities, 1.0 - probabilities)
            if self.mode == "asymmetric_focal":
                gamma = torch.where(
                    hard_labels >= 0.5,
                    torch.full_like(loss, self.focal_pos_gamma),
                    torch.full_like(loss, self.focal_neg_gamma),
                )
            else:
                gamma = torch.full_like(loss, self.focal_gamma)
            loss = loss * torch.clamp(1.0 - pt, min=1e-6).pow(gamma)

        return (loss * weights).mean()


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, shadow_value in self.shadow.items():
            shadow_value.mul_(self.decay).add_(state[key].detach(), alpha=1.0 - self.decay)

    def store(self, model: nn.Module) -> None:
        self.backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, shadow_value in self.shadow.items():
            state[key].copy_(shadow_value)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in self.backup.items():
            state[key].copy_(value)
        self.backup = {}


@contextmanager
def ema_scope(model: nn.Module, ema: ModelEma | None):
    if ema is None:
        yield
        return
    ema.store(model)
    ema.copy_to(model)
    try:
        yield
    finally:
        ema.restore(model)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if args.scheduler == "none":
        return None

    warmup_epochs = max(int(args.warmup_epochs), 0)
    total_epochs = max(int(args.epochs), 1)
    min_factor = args.min_lr / max(args.lr, 1e-12)

    def schedule(epoch_index: int) -> float:
        epoch = epoch_index + 1
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return max(epoch / warmup_epochs, min_factor)
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int | None,
    grad_clip: float,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    ema: ModelEma | None,
) -> float:
    model.train()
    running_loss = 0.0
    seen_batches = 0
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)

    for batch_index, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        running_loss += float(loss.item())
        seen_batches += 1
        progress.set_postfix(loss=f"{running_loss / seen_batches:.4f}")

        if max_batches is not None and batch_index >= max_batches:
            break

    return running_loss / max(seen_batches, 1)


@torch.no_grad()
def evaluate_liveness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_apcer: float,
    ema: ModelEma | None = None,
) -> tuple[float, LivenessMetrics, np.ndarray, np.ndarray]:
    with ema_scope(model, ema):
        model.eval()
        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for images, batch_labels in loader:
            images = images.to(device)
            probabilities = torch.sigmoid(model(images)).detach().cpu().numpy().reshape(-1)
            scores.append(probabilities)
            labels.append(batch_labels.numpy().reshape(-1))

    all_scores = np.concatenate(scores)
    all_labels = np.concatenate(labels).astype(np.int64)
    auc = binary_roc_auc(all_scores, all_labels)
    metrics = select_threshold(all_scores, all_labels, max_apcer=max_apcer)
    return auc, metrics, all_scores, all_labels


def targets_met(auc: float, metrics: LivenessMetrics, args: argparse.Namespace) -> bool:
    return (
        auc >= args.target_roc_auc
        and metrics.acer <= args.target_acer
        and metrics.apcer <= args.target_apcer
        and metrics.bpcer <= args.target_bpcer
        and metrics.accuracy >= args.target_accuracy
    )


def ranking_score(auc: float, metrics: LivenessMetrics, args: argparse.Namespace) -> tuple[float, float, float, float]:
    apcer_overage = max(metrics.apcer - args.target_apcer, 0.0)
    return (apcer_overage, metrics.acer, metrics.bpcer, -auc)


def mine_hard_false_accepts(
    model: nn.Module,
    loader: DataLoader,
    dataset: LivenessImageDataset,
    device: torch.device,
    threshold: float,
    *,
    max_items: int,
    ema: ModelEma | None,
) -> list[LivenessSample]:
    with ema_scope(model, ema):
        model.eval()
        mined: list[tuple[float, int]] = []
        offset = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                scores = torch.sigmoid(model(images)).detach().cpu().numpy().reshape(-1)
                label_values = labels.numpy().reshape(-1).astype(np.int64)
                for local_index, (score, label) in enumerate(zip(scores, label_values, strict=True)):
                    if label == 0 and float(score) >= threshold:
                        mined.append((float(score), offset + local_index))
                offset += len(label_values)
    mined.sort(reverse=True)
    selected: list[LivenessSample] = []
    for _, index in mined[:max_items]:
        selected.append(dataset.samples[index % len(dataset.samples)])
    return selected


def init_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "lr",
                "roc_auc",
                "accuracy",
                "balanced_accuracy",
                "apcer",
                "bpcer",
                "acer",
                "threshold",
                "targets_met",
            ],
        )
        writer.writeheader()


def append_metrics_csv(
    path: Path,
    epoch: int,
    train_loss: float,
    lr: float,
    auc: float,
    metrics: LivenessMetrics,
    passed: bool,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "lr",
                "roc_auc",
                "accuracy",
                "balanced_accuracy",
                "apcer",
                "bpcer",
                "acer",
                "threshold",
                "targets_met",
            ],
        )
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "lr": f"{lr:.8f}",
                "roc_auc": f"{auc:.6f}",
                "accuracy": f"{metrics.accuracy:.6f}",
                "balanced_accuracy": f"{metrics.balanced_accuracy:.6f}",
                "apcer": f"{metrics.apcer:.6f}",
                "bpcer": f"{metrics.bpcer:.6f}",
                "acer": f"{metrics.acer:.6f}",
                "threshold": f"{metrics.threshold:.6f}",
                "targets_met": str(passed).lower(),
            }
        )


def checkpoint_config(args: argparse.Namespace, threshold: float) -> dict[str, object]:
    return {
        "architecture": args.architecture,
        "input_size": args.image_size,
        "width_mult": args.width_mult,
        "dropout": args.dropout,
        "normalization": args.normalization,
        "threshold": float(threshold),
        "optimizer": args.optimizer,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "loss": args.loss,
        "live_loss_weight": args.live_loss_weight,
        "spoof_loss_weight": args.spoof_loss_weight,
        "focal_gamma": args.focal_gamma,
        "focal_pos_gamma": args.focal_pos_gamma,
        "focal_neg_gamma": args.focal_neg_gamma,
        "label_smoothing": args.label_smoothing,
        "balanced_sampler": args.balanced_sampler,
        "ema": args.ema,
        "ema_decay": args.ema_decay,
        "augmentation_preset": "none" if args.no_augment else args.augmentation_preset,
        "augmentation_config": {} if args.no_augment else get_augmentation_config(args.augmentation_preset),
        "seed": args.seed,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    train_loss: float,
    args: argparse.Namespace,
    auc: float | None = None,
    metrics: LivenessMetrics | None = None,
    ema: ModelEma | None = None,
    scores: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold = metrics.threshold if metrics is not None else args.default_threshold
    with ema_scope(model, ema):
        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint: dict[str, object] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "config": checkpoint_config(args, threshold),
        "model_state_dict": state_dict,
    }
    if metrics is not None and auc is not None:
        checkpoint["metrics"] = {
            "roc_auc": auc,
            "accuracy": metrics.accuracy,
            "balanced_accuracy": metrics.balanced_accuracy,
            "apcer": metrics.apcer,
            "bpcer": metrics.bpcer,
            "acer": metrics.acer,
            "threshold": metrics.threshold,
        }
    torch.save(checkpoint, path)

    threshold_payload: dict[str, object]
    if scores is not None and labels is not None:
        selected = choose_security_threshold(scores, labels)
        reject_threshold = min(float(selected["threshold"]), metrics.threshold if metrics else float(selected["threshold"]))
        threshold_payload = {
            "live_label": 1,
            "threshold": float(selected["threshold"]),
            "policy": selected["policy"],
            "target_apcer": selected["target_apcer"],
            "decision_policy": {
                "reject_threshold": float(reject_threshold),
                "accept_threshold": float(selected["threshold"]),
                "threshold_policy": selected["policy"],
            },
        }
    else:
        threshold_payload = {
            "threshold": float(threshold),
            "live_label": 1,
            "decision_policy": {
                "reject_threshold": float(threshold),
                "accept_threshold": float(threshold),
                "threshold_policy": "checkpoint_threshold",
            },
        }
    path.with_suffix(".threshold.json").write_text(
        json.dumps(threshold_payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TinyLiveness.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/liveness"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--architecture", choices=MODEL_CHOICES, default="tiny")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--width-mult", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--normalization",
        choices=["tinyliveness", "imagenet"],
        default="tinyliveness",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "weighted_bce", "focal", "asymmetric_focal"],
        default="bce",
    )
    parser.add_argument("--live-loss-weight", type=float, default=1.0)
    parser.add_argument("--spoof-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-pos-gamma", type=float, default=1.0)
    parser.add_argument("--focal-neg-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--train-repeats", type=int, default=1)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--augmentation-preset",
        choices=["light", "medium", "strong", "spoof_artifact_strong"],
        default="medium",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-apcer", type=float, default=0.03)
    parser.add_argument("--target-roc-auc", type=float, default=0.98)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--target-apcer", type=float, default=0.03)
    parser.add_argument("--target-bpcer", type=float, default=0.10)
    parser.add_argument("--target-acer", type=float, default=0.05)
    parser.add_argument("--stop-when-target-met", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--hard-negative-mining", action="store_true")
    parser.add_argument("--hard-negative-after-epoch", type=int, default=2)
    parser.add_argument("--hard-negative-max", type=int, default=512)
    parser.add_argument("--hard-negative-repeat", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tinyliveness.pt"))
    parser.add_argument("--best-output", type=Path, default=Path("checkpoints/best_tinyliveness.pt"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("checkpoints/tinyliveness_metrics.csv"))
    parser.add_argument("--max-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_reproducible_seed(args.seed)
    device = pick_device(args.device)

    train_dataset = LivenessImageDataset(
        args.data_dir / args.train_split,
        image_size=args.image_size,
        augment=not args.no_augment,
        repeats=args.train_repeats,
        normalization=args.normalization,
        augmentation_preset=args.augmentation_preset,
    )
    val_dataset = LivenessImageDataset(
        args.data_dir / args.val_split,
        image_size=args.image_size,
        augment=False,
        normalization=args.normalization,
    )
    val_loader = make_loader(val_dataset, args, device, training=False)

    model = build_liveness_model(
        args.architecture,
        width_mult=args.width_mult,
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    if args.init_checkpoint is not None:
        load_initial_checkpoint(model, args.init_checkpoint, device)
    loss_fn = BinaryLivenessLoss(
        mode=args.loss,
        live_weight=args.live_loss_weight,
        spoof_weight=args.spoof_loss_weight,
        focal_gamma=args.focal_gamma,
        focal_pos_gamma=args.focal_pos_gamma,
        focal_neg_gamma=args.focal_neg_gamma,
        label_smoothing=args.label_smoothing,
        dataset=train_dataset,
        balanced_sampler=args.balanced_sampler,
    )

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = build_scheduler(optimizer, args)
    ema = ModelEma(model, decay=args.ema_decay) if args.ema else None

    train_counts = train_dataset.class_counts()
    val_counts = val_dataset.class_counts()
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"input size: {args.image_size}x{args.image_size}")
    print(f"architecture: {args.architecture}")
    print(f"pretrained: {args.pretrained}")
    print(f"normalization: {args.normalization}")
    print(f"width multiplier: {args.width_mult}")
    print(f"model params: {count_parameters(model):,}")
    print(f"estimated fp32 model size: {count_parameters(model) * 4 / (1024 * 1024):.2f} MB")
    print(f"train samples: {len(train_dataset)} live={train_counts['live']} spoof={train_counts['spoof']}")
    print(f"val samples: {len(val_dataset)} live={val_counts['live']} spoof={val_counts['spoof']}")
    print(f"augmentation: {not args.no_augment} preset={args.augmentation_preset if not args.no_augment else 'none'}")
    print(f"balanced sampler: {args.balanced_sampler}")
    print(f"loss: {args.loss} live_weight={args.live_loss_weight} spoof_weight={args.spoof_loss_weight}")
    print(f"optimizer: {args.optimizer}")
    print(f"scheduler: {args.scheduler} warmup_epochs={args.warmup_epochs}")
    print(f"ema: {args.ema} decay={args.ema_decay}")
    print(f"mixed precision: {args.amp and device.type == 'cuda'}")
    print(f"threshold policy: minimize ACER with APCER <= {args.max_apcer:.4f}")

    init_metrics_csv(args.metrics_csv)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda")
    last_loss = 0.0
    best_score: tuple[float, float, float, float] | None = None
    best_auc: float | None = None
    best_metrics: LivenessMetrics | None = None
    best_scores: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    epochs_without_improvement = 0
    hard_negative_paths: set[Path] = set()

    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(train_dataset, args, device, training=True)
        last_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=args.max_batches,
            grad_clip=args.grad_clip,
            use_amp=args.amp and device.type == "cuda",
            scaler=scaler,
            ema=ema,
        )
        print(f"epoch {epoch:03d} train_loss={last_loss:.4f}")

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            auc, metrics, scores, labels = evaluate_liveness(
                model=model,
                loader=val_loader,
                device=device,
                max_apcer=args.max_apcer,
                ema=ema,
            )
            passed = targets_met(auc, metrics, args)
            print(
                "epoch "
                f"{epoch:03d} val_auc={auc:.4f} "
                f"acc={metrics.accuracy:.4f} "
                f"bal_acc={metrics.balanced_accuracy:.4f} "
                f"apcer={metrics.apcer:.4f} "
                f"bpcer={metrics.bpcer:.4f} "
                f"acer={metrics.acer:.4f} "
                f"threshold={metrics.threshold:.4f} "
                f"targets={'PASS' if passed else 'FAIL'}"
            )
            append_metrics_csv(
                path=args.metrics_csv,
                epoch=epoch,
                train_loss=last_loss,
                lr=optimizer.param_groups[0]["lr"],
                auc=auc,
                metrics=metrics,
                passed=passed,
            )

            score = ranking_score(auc, metrics, args)
            if best_score is None or score < best_score:
                best_score = score
                best_auc = auc
                best_metrics = metrics
                best_scores = scores
                best_labels = labels
                epochs_without_improvement = 0
                save_checkpoint(
                    path=args.best_output,
                    model=model,
                    epoch=epoch,
                    train_loss=last_loss,
                    args=args,
                    auc=auc,
                    metrics=metrics,
                    ema=ema,
                    scores=scores,
                    labels=labels,
                )
                print(f"saved best checkpoint: {args.best_output}")
            else:
                epochs_without_improvement += 1

            if args.hard_negative_mining and epoch >= args.hard_negative_after_epoch:
                mined = mine_hard_false_accepts(
                    model,
                    val_loader,
                    val_dataset,
                    device,
                    metrics.threshold,
                    max_items=args.hard_negative_max,
                    ema=ema,
                )
                new_samples: list[LivenessSample] = []
                for sample in mined:
                    if sample.path in hard_negative_paths:
                        continue
                    hard_negative_paths.add(sample.path)
                    for _ in range(max(args.hard_negative_repeat, 1)):
                        new_samples.append(sample)
                if new_samples:
                    train_dataset.add_samples(new_samples)
                    print(f"added hard false accepts from {args.val_split}: {len(new_samples)}")

            if passed and args.stop_when_target_met:
                print("all target metrics met; stopping early")
                break
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"early stopping after {epochs_without_improvement} evals without improvement")
                break

        if scheduler is not None:
            scheduler.step()

    save_checkpoint(
        path=args.output,
        model=model,
        epoch=args.epochs,
        train_loss=last_loss,
        args=args,
        auc=best_auc,
        metrics=best_metrics,
        ema=ema,
        scores=best_scores,
        labels=best_labels,
    )
    print(f"saved checkpoint: {args.output}")
    if best_metrics is not None:
        print(f"best val auc: {best_auc:.4f}")
        print(f"best val acer: {best_metrics.acer:.4f}")
        print(f"best threshold: {best_metrics.threshold:.4f}")
        print(f"best checkpoint: {args.best_output}")


if __name__ == "__main__":
    main()
