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

from utils.true_dataset import create_dataloaders
import utils.true_dataset as true_dataset_module
from models.model import CrossTransVFC, MMConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.argmax(dim=-1)


def build_loss(device: torch.device, class_weights: Optional[torch.Tensor] = None):
    return nn.CrossEntropyLoss(weight=class_weights).to(device)


def compute_class_weights(dataset, num_classes: int, device: torch.device):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for sample in dataset:
        label_idx = int(sample["label"].argmax().item())
        if 0 <= label_idx < num_classes:
            counts[label_idx] += 1.0

    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device), counts


@torch.no_grad()
def evaluate(model, loader, device, loss_func, desc="Evaluating") -> Dict[str, Any]:
    """Evaluate model on a dataloader."""
    model.eval()

    all_y, all_p = [], []
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc=desc):
        labels = normalize_labels(batch["label"]).to(device)

        with autocast(device_type="cuda"):
            out = model(
                claim=batch["claim"],
                text_evidence=batch["content"],
                image_evidence=batch["keyframes"],
                labels=labels,
            )
            logits = out["logits"]
            loss = loss_func(logits, labels)

        preds = logits.argmax(dim=-1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size

        all_y.extend(labels.cpu().tolist())
        all_p.extend(preds.cpu().tolist())

    acc = accuracy_score(all_y, all_p)
    f1m = f1_score(all_y, all_p, average="macro", zero_division=0)
    avg_loss = total_loss / max(n, 1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1m,
        "y_true": all_y,
        "y_pred": all_p,
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
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader, desc=f"Training {ep}/{epochs}"):
        labels = normalize_labels(batch["label"]).to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            out = model(
                claim=batch["claim"],
                text_evidence=batch["content"],
                image_evidence=batch["keyframes"],
                labels=labels,
            )
            logits = out["logits"]
            loss = loss_func(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


def main(args):
    DATA_ROOT = Path("./data/TRUE_Dataset")
    print(f"Using dataset module: {true_dataset_module.__file__}")

    # ====== Training hyperparams ======
    seed = 42
    batch_size = args.batch_size
    epochs = 30
    lr = 3e-5
    warmup_ratio = 0.06
    num_workers = 8
    patience = 5

    cfg = MMConfig(
        _claim_pt=args.text_model,
        _long_pt=args.long_text_model,
        _video_pt=args.video_model,
        _vision_pt=args.image_model,
        num_classes=2,
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

    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        path=str(DATA_ROOT),
        batch_size=batch_size,
        shuffle_train=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Class-balanced sampling for training batches.
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

    label2id = {"TRUE": 0, "FALSE": 1}
    id2label = {v: k for k, v in label2id.items()}

    model = CrossTransVFC(cfg)
    model = model.to(device)

    class_weights, class_counts = compute_class_weights(
        train_loader.dataset, cfg.num_classes, device
    )
    print(
        f"Class counts (train): TRUE={int(class_counts[0].item())}, "
        f"FALSE={int(class_counts[1].item())}"
    )
    print(f"Class weights (CE): {class_weights.tolist()}")
    loss_func = build_loss(device, class_weights=class_weights)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
    )

    scaler = GradScaler("cuda")

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = -1.0
    best_acc = -1.0
    patience_counter = 0

    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.saved_prefix:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dir_name = f"run_{timestamp}"
    else:
        dir_name = f"{args.saved_prefix}"
    run_dir = save_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoints will be saved to: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        config_dict = {
            "seed": seed,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "warmup_ratio": warmup_ratio,
            "model_config": cfg.__dict__,
            "label2id": label2id,
        }
        json.dump(config_dict, f, indent=2)

    train_log_path = run_dir / "train_log.csv"
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Acc,Val_F1\n")

    print(f"\n{'=' * 70}")
    print("Starting Training")
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
            f"val_acc={metrics['acc']:.4f} | "
            f"val_f1={metrics['f1_macro']:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.2e}")

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},{tr_loss:.4f},{metrics['loss']:.4f},{metrics['acc']:.4f},{metrics['f1_macro']:.4f}\n"
            )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_acc = metrics["acc"]
            patience_counter = 0
            ckpt = {
                "epoch": ep,
                "cfg": cfg.__dict__,
                "label2id": label2id,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "best_acc": best_acc,
                "val_metrics": metrics,
            }
            best_checkpoint = run_dir / "best.pt"
            torch.save(ckpt, best_checkpoint)
            print(f"✓ Saved best model to {best_checkpoint}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

        if ep % 5 == 0:
            latest_checkpoint = run_dir / f"epoch_{ep:03d}.pt"
            ckpt = {
                "epoch": ep,
                "cfg": cfg.__dict__,
                "label2id": label2id,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt, latest_checkpoint)
            print(f"Saved checkpoint to {latest_checkpoint}")

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
        f"Best Val F1: {checkpoint['best_f1']:.4f}, Best Val Acc: {checkpoint['best_acc']:.4f}"
    )

    # Test
    test_metrics = evaluate(model, test_loader, device, loss_func, desc="Testing")

    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]

    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['acc']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")

    print("\nDetailed Classification Report:")
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(label2id))],
        digits=4,
        zero_division=0,
    )
    print(clf_report)

    with open(run_dir / "test_log.txt", "w", encoding="utf-8") as f:
        f.write("Testing with Best Model\n")
        f.write(f"Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Accuracy: {test_metrics['acc']:.4f}\n")
        f.write(f"F1 (macro): {test_metrics['f1_macro']:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(clf_report + "\n")

    test_results = {
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["acc"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
        "best_val_f1": float(best_f1),
        "best_val_acc": float(best_acc),
        "predictions": y_pred,
        "true_labels": y_true,
    }

    with open(run_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✓ Test results saved to {run_dir / 'test_results.json'}")
    print(f"✓ Training completed! All outputs saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--text_model", type=str, default="roberta-base")
    parser.add_argument("--long_text_model", type=str, default="longformer")
    parser.add_argument("--image_model", type=str, default="clip")
    parser.add_argument("--video_model", type=str, default="videomae")
    parser.add_argument("--saved-prefix", type=str, default="cross_trans_vfc")
    args = parser.parse_args()
    main(args)
