"""
Hierarchical multi-class training for CrossTransVFC.

Two-stage approach:
  1. Load pre-trained binary CrossTransVFC checkpoint (optional)
  2. Fine-tune with joint coarse + fine-grained loss

Usage:
    # From scratch
    python train_multiclass.py --batch_size 32

    # From pre-trained binary model
    python train_multiclass.py --pretrained_ckpt checkpoints/cross_trans_vfc/best.pt

    # Freeze backbone, only train classifier heads
    python train_multiclass.py --pretrained_ckpt checkpoints/cross_trans_vfc/best.pt --freeze_backbone
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import get_cosine_schedule_with_warmup

from utils.true_dataset import (
    create_dataloaders,
    NUM_FINE_PER_COARSE,
    TOTAL_FINE_CLASSES,
    RATING_TO_FINE,
    RATING_TO_FLAT_FINE,
)
import utils.true_dataset as true_dataset_module
from models.model import MMConfig
from models.model_multiclass import HierarchicalCrossTransVFC

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Label maps ──
COARSE_LABELS = {0: "TRUE", 1: "FALSE"}

FINE_LABELS = {
    0: "true",
    1: "mostly_true",
    2: "correct_attribution",
    3: "false",
    4: "mostly_false",
    5: "mixture",
    6: "fake",
    7: "miscaptioned",
}


def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    """Convert one-hot binary labels to class indices."""
    return labels.argmax(dim=-1)


def build_hierarchical_loss(
    device: torch.device,
    coarse_weights: Optional[torch.Tensor] = None,
    fine_weights: Optional[torch.Tensor] = None,
    lambda_coarse: float = 1.0,
    lambda_fine: float = 1.0,
):
    """Build a combined coarse + fine CE loss function."""
    coarse_ce = nn.CrossEntropyLoss(weight=coarse_weights).to(device)

    # Split fine weights for each coarse group
    fine_ce_funcs = []
    offset = 0
    for n_fine in NUM_FINE_PER_COARSE:
        w_c = (
            fine_weights[offset : offset + n_fine] if fine_weights is not None else None
        )
        fine_ce_funcs.append(
            nn.CrossEntropyLoss(weight=w_c, ignore_index=-1).to(device)
        )
        offset += n_fine

    def hierarchical_loss(coarse_logits, fine_logits, coarse_labels, fine_labels):
        loss_coarse = coarse_ce(coarse_logits, coarse_labels)

        # For fine loss, compute per-group CE
        loss_fine = torch.tensor(0.0, device=device)
        n_fine_samples = 0

        for c in range(2):
            mask_c = coarse_labels == c
            if not mask_c.any():
                continue
            n_fine_c = NUM_FINE_PER_COARSE[c]
            # fine_logits[mask_c] has shape (N_c, max_fine)
            # Only use first n_fine_c columns
            logits_c = fine_logits[mask_c, :n_fine_c]
            targets_c = fine_labels[mask_c]
            loss_fine = loss_fine + fine_ce_funcs[c](logits_c, targets_c) * mask_c.sum()
            n_fine_samples += mask_c.sum()

        if n_fine_samples > 0:
            loss_fine = loss_fine / n_fine_samples

        return (
            lambda_coarse * loss_coarse + lambda_fine * loss_fine,
            loss_coarse,
            loss_fine,
        )

    return hierarchical_loss


def compute_class_weights(dataset, num_classes: int, device: torch.device):
    """Compute inverse-frequency class weights for binary labels."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for sample in dataset:
        label_idx = int(sample["label"].argmax().item())
        if 0 <= label_idx < num_classes:
            counts[label_idx] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device), counts


def compute_fine_class_weights(dataset, device: torch.device):
    """Compute inverse-frequency weights for flat fine-grained labels."""
    total = TOTAL_FINE_CLASSES
    counts = torch.zeros(total, dtype=torch.float32)
    for sample in dataset:
        flat_idx = sample["flat_fine_label"]
        if 0 <= flat_idx < total:
            counts[flat_idx] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (total * counts)
    return weights.to(device), counts


@torch.no_grad()
def evaluate(model, loader, device, loss_func, desc="Evaluating") -> Dict[str, Any]:
    """Evaluate model with hierarchical metrics."""
    model.eval()

    all_coarse_true, all_coarse_pred = [], []
    all_fine_true, all_fine_pred = [], []
    all_flat_fine_true, all_flat_fine_pred = [], []
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        coarse_labels = normalize_labels(batch["label"]).to(device)
        fine_labels = batch["fine_label"].to(device)
        flat_fine_labels = batch["flat_fine_label"].to(device)

        with autocast(device_type="cuda"):
            out = model(
                claim=batch["claim"],
                text_evidence=batch["content"],
                image_evidence=batch["keyframes"],
                coarse_labels=coarse_labels,
            )
            loss, _, _ = loss_func(
                out["coarse_logits"],
                out["fine_logits"],
                coarse_labels,
                fine_labels,
            )

        coarse_preds = out["coarse_logits"].argmax(dim=-1)

        # Fine predictions: for each sample, get fine pred from its coarse group
        fine_preds = []
        for i in range(coarse_labels.size(0)):
            c = coarse_preds[i].item()
            n_fine_c = NUM_FINE_PER_COARSE[c]
            fine_pred_i = out["fine_logits"][i, :n_fine_c].argmax().item()
            fine_preds.append(fine_pred_i)

        # Flat fine prediction: offset by coarse group
        flat_fine_preds = []
        for i in range(coarse_labels.size(0)):
            c = coarse_preds[i].item()
            n_fine_c = NUM_FINE_PER_COARSE[c]
            fine_pred_i = out["fine_logits"][i, :n_fine_c].argmax().item()
            offset = sum(NUM_FINE_PER_COARSE[:c])
            flat_fine_preds.append(offset + fine_pred_i)

        batch_size = coarse_labels.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size

        all_coarse_true.extend(coarse_labels.cpu().tolist())
        all_coarse_pred.extend(coarse_preds.cpu().tolist())
        all_fine_true.extend(fine_labels.cpu().tolist())
        all_fine_pred.extend(fine_preds)
        all_flat_fine_true.extend(flat_fine_labels.cpu().tolist())
        all_flat_fine_pred.extend(flat_fine_preds)

    coarse_acc = accuracy_score(all_coarse_true, all_coarse_pred)
    coarse_f1 = f1_score(
        all_coarse_true, all_coarse_pred, average="macro", zero_division=0
    )
    flat_fine_acc = accuracy_score(all_flat_fine_true, all_flat_fine_pred)
    flat_fine_f1 = f1_score(
        all_flat_fine_true, all_flat_fine_pred, average="macro", zero_division=0
    )

    # Hierarchical consistency: fine pred consistent with coarse pred
    consistent = sum(1 for ct, cp in zip(all_coarse_true, all_coarse_pred) if ct == cp)
    consistency = consistent / max(len(all_coarse_true), 1)

    return {
        "loss": total_loss / max(n, 1),
        "coarse_acc": coarse_acc,
        "coarse_f1": coarse_f1,
        "flat_fine_acc": flat_fine_acc,
        "flat_fine_f1": flat_fine_f1,
        "consistency": consistency,
        "coarse_true": all_coarse_true,
        "coarse_pred": all_coarse_pred,
        "flat_fine_true": all_flat_fine_true,
        "flat_fine_pred": all_flat_fine_pred,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    ep,
    epochs,
    device,
    loss_func,
    grad_clip=1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc=f"Training {ep}/{epochs}"):
        coarse_labels = normalize_labels(batch["label"]).to(device)
        fine_labels = batch["fine_label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            out = model(
                claim=batch["claim"],
                text_evidence=batch["content"],
                image_evidence=batch["keyframes"],
                coarse_labels=coarse_labels,
            )
            loss, _, _ = loss_func(
                out["coarse_logits"],
                out["fine_logits"],
                coarse_labels,
                fine_labels,
            )

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        bs = coarse_labels.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


def main(args):
    DATA_ROOT = Path("./data/TRUE_Dataset")
    print(f"Using dataset module: {true_dataset_module.__file__}")

    # ====== Hyperparams ======
    seed = 42
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    warmup_ratio = 0.06
    num_workers = 8
    patience = args.patience

    cfg = MMConfig(
        _claim_pt=args.text_model,
        _long_pt=args.long_text_model,
        _video_pt=args.video_model,
        _vision_pt=args.image_model,
        num_classes=2,  # coarse classes (binary)
        freeze_text=True,
        freeze_long_text=True,
        freeze_vision=True,
        freeze_video=True,
        unfreeze_text_last_n=2,
        unfreeze_long_last_n=1,
        cls_dropout=0.2,
        mfm_dropout=0.2,
        claim_max_length=256,
        evidence_max_length=512,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====== Data ======
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        path=str(DATA_ROOT),
        batch_size=batch_size,
        shuffle_train=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Weighted sampler for training (based on coarse labels)
    train_dataset = train_loader.dataset
    sample_labels = torch.tensor(
        [int(sample["label"].argmax().item()) for sample in train_dataset],
        dtype=torch.long,
    )
    class_counts_sampler = torch.bincount(
        sample_labels, minlength=cfg.num_classes
    ).float()
    class_counts_sampler = class_counts_sampler.clamp_min(1.0)
    sample_class_weights = class_counts_sampler.sum() / (
        cfg.num_classes * class_counts_sampler
    )
    sample_weights = sample_class_weights[sample_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=train_loader.collate_fn,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(
        f"Dataset sizes: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    # ====== Model ======
    model = HierarchicalCrossTransVFC(cfg, num_fine_per_coarse=NUM_FINE_PER_COARSE)

    # Load pre-trained binary weights if provided
    if args.pretrained_ckpt:
        model.load_pretrained_binary(args.pretrained_ckpt, device)

    model = model.to(device)

    # Optionally freeze backbone
    if args.freeze_backbone:
        print("Freezing backbone — only training classifier heads")
        for name, param in model.named_parameters():
            if "hierarchical_classifier" not in name:
                param.requires_grad = False

    # Class weights
    coarse_weights, coarse_counts = compute_class_weights(
        train_loader.dataset, cfg.num_classes, device
    )
    fine_weights, fine_counts = compute_fine_class_weights(train_loader.dataset, device)

    print(
        f"\nCoarse class counts: TRUE={int(coarse_counts[0])}, FALSE={int(coarse_counts[1])}"
    )
    print(f"Fine class counts: {fine_counts.int().tolist()}")

    loss_func = build_hierarchical_loss(
        device,
        coarse_weights=coarse_weights,
        fine_weights=fine_weights,
        lambda_coarse=args.lambda_coarse,
        lambda_fine=args.lambda_fine,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    scaler = GradScaler("cuda")

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_fine_f1 = -1.0
    best_coarse_f1 = -1.0
    patience_counter = 0

    # ====== Save dir ======
    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.saved_prefix:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dir_name = f"multiclass_{timestamp}"
    else:
        dir_name = f"{args.saved_prefix}"
    run_dir = save_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints will be saved to: {run_dir}")

    # Save config
    with open(run_dir / "config.json", "w") as f:
        config_dict = {
            "seed": seed,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "warmup_ratio": warmup_ratio,
            "lambda_coarse": args.lambda_coarse,
            "lambda_fine": args.lambda_fine,
            "freeze_backbone": args.freeze_backbone,
            "pretrained_ckpt": args.pretrained_ckpt,
            "model_config": cfg.__dict__,
            "num_fine_per_coarse": list(NUM_FINE_PER_COARSE),
            "coarse_labels": COARSE_LABELS,
            "fine_labels": FINE_LABELS,
        }
        json.dump(config_dict, f, indent=2)

    train_log_path = run_dir / "train_log.csv"
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write(
            "Epoch,Train_Loss,Val_Loss,Val_Coarse_Acc,Val_Coarse_F1,Val_Fine_Acc,Val_Fine_F1\n"
        )

    # ====== Training loop ======
    print(f"\n{'=' * 70}")
    print("Starting Hierarchical Multi-class Training")
    print(f"{'=' * 70}\n")

    for ep in range(1, epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {ep}/{epochs}")
        print(f"{'=' * 70}")

        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            ep,
            epochs,
            device,
            loss_func,
            grad_clip=1.0,
        )
        metrics = evaluate(model, val_loader, device, loss_func)

        print(
            f"\n[Epoch {ep}/{epochs}] "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"coarse_acc={metrics['coarse_acc']:.4f} | "
            f"coarse_f1={metrics['coarse_f1']:.4f} | "
            f"fine_acc={metrics['flat_fine_acc']:.4f} | "
            f"fine_f1={metrics['flat_fine_f1']:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.2e}")

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},{tr_loss:.4f},{metrics['loss']:.4f},"
                f"{metrics['coarse_acc']:.4f},{metrics['coarse_f1']:.4f},"
                f"{metrics['flat_fine_acc']:.4f},{metrics['flat_fine_f1']:.4f}\n"
            )

        # Early stopping on fine-grained F1
        if metrics["flat_fine_f1"] > best_fine_f1:
            best_fine_f1 = metrics["flat_fine_f1"]
            best_coarse_f1 = metrics["coarse_f1"]
            patience_counter = 0
            ckpt = {
                "epoch": ep,
                "cfg": cfg.__dict__,
                "num_fine_per_coarse": list(NUM_FINE_PER_COARSE),
                "coarse_labels": COARSE_LABELS,
                "fine_labels": FINE_LABELS,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_fine_f1": best_fine_f1,
                "best_coarse_f1": best_coarse_f1,
            }
            best_checkpoint = run_dir / "best.pt"
            torch.save(ckpt, best_checkpoint)
            print(f"✓ Saved best model to {best_checkpoint}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

        if ep % 5 == 0:
            epoch_checkpoint = run_dir / f"epoch_{ep:03d}.pt"
            ckpt = {
                "epoch": ep,
                "cfg": cfg.__dict__,
                "num_fine_per_coarse": list(NUM_FINE_PER_COARSE),
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt, epoch_checkpoint)
            print(f"Saved checkpoint to {epoch_checkpoint}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {ep} epochs")
            break

    # ====== Final Testing ======
    print(f"\n{'=' * 70}")
    print("Testing with Best Model")
    print(f"{'=' * 70}")

    best_checkpoint = run_dir / "best.pt"
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(
        f"Best Val Fine F1: {checkpoint['best_fine_f1']:.4f}, "
        f"Best Val Coarse F1: {checkpoint['best_coarse_f1']:.4f}"
    )

    test_metrics = evaluate(model, test_loader, device, loss_func, desc="Testing")

    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Coarse Accuracy: {test_metrics['coarse_acc']:.4f}")
    print(f"  Coarse F1 (macro): {test_metrics['coarse_f1']:.4f}")
    print(f"  Fine Accuracy: {test_metrics['flat_fine_acc']:.4f}")
    print(f"  Fine F1 (macro): {test_metrics['flat_fine_f1']:.4f}")
    print(f"  Hierarchical Consistency: {test_metrics['consistency']:.4f}")

    # Coarse classification report
    print("\nCoarse Classification Report:")
    coarse_report = classification_report(
        test_metrics["coarse_true"],
        test_metrics["coarse_pred"],
        target_names=[COARSE_LABELS[i] for i in range(2)],
        digits=4,
        zero_division=0,
    )
    print(coarse_report)

    # Fine classification report
    print("\nFine-grained Classification Report:")
    fine_report = classification_report(
        test_metrics["flat_fine_true"],
        test_metrics["flat_fine_pred"],
        target_names=[FINE_LABELS[i] for i in range(TOTAL_FINE_CLASSES)],
        digits=4,
        zero_division=0,
    )
    print(fine_report)

    # Save test results
    with open(run_dir / "test_log.txt", "w", encoding="utf-8") as f:
        f.write("Testing with Best Hierarchical Model\n")
        f.write(f"Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Coarse Accuracy: {test_metrics['coarse_acc']:.4f}\n")
        f.write(f"Coarse F1 (macro): {test_metrics['coarse_f1']:.4f}\n")
        f.write(f"Fine Accuracy: {test_metrics['flat_fine_acc']:.4f}\n")
        f.write(f"Fine F1 (macro): {test_metrics['flat_fine_f1']:.4f}\n")
        f.write(f"Hierarchical Consistency: {test_metrics['consistency']:.4f}\n\n")
        f.write("Coarse Classification Report:\n")
        f.write(coarse_report + "\n")
        f.write("Fine-grained Classification Report:\n")
        f.write(fine_report + "\n")

    test_results = {
        "test_loss": float(test_metrics["loss"]),
        "coarse_accuracy": float(test_metrics["coarse_acc"]),
        "coarse_f1_macro": float(test_metrics["coarse_f1"]),
        "fine_accuracy": float(test_metrics["flat_fine_acc"]),
        "fine_f1_macro": float(test_metrics["flat_fine_f1"]),
        "hierarchical_consistency": float(test_metrics["consistency"]),
        "best_val_fine_f1": float(best_fine_f1),
        "best_val_coarse_f1": float(best_coarse_f1),
        "coarse_predictions": test_metrics["coarse_pred"],
        "coarse_true_labels": test_metrics["coarse_true"],
        "fine_predictions": test_metrics["flat_fine_pred"],
        "fine_true_labels": test_metrics["flat_fine_true"],
    }

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✓ Test results saved to {run_dir / 'test_results.json'}")
    print(f"✓ Training completed! All outputs saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical multi-class training for CrossTransVFC"
    )
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--lambda_coarse",
        type=float,
        default=1.0,
        help="Weight for coarse (binary) loss",
    )
    parser.add_argument(
        "--lambda_fine", type=float, default=1.0, help="Weight for fine-grained loss"
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="",
        help="Path to pre-trained binary CrossTransVFC checkpoint",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone, only train hierarchical heads",
    )
    parser.add_argument("--text_model", type=str, default="roberta-base")
    parser.add_argument("--long_text_model", type=str, default="longformer")
    parser.add_argument("--image_model", type=str, default="clip")
    parser.add_argument("--video_model", type=str, default="videomae")
    parser.add_argument("--saved-prefix", type=str, default="hierarchical_multiclass")
    args = parser.parse_args()
    main(args)
