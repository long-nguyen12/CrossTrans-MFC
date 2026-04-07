"""
Evaluate a trained hierarchical multi-class ablation study checkpoint for CrossTransVFC.

Re-constructs the specific ablation variant Architecture and reports metrics at two levels:
  - Coarse: binary TRUE/FALSE accuracy, F1, precision, recall
  - Fine-grained: 8-class accuracy, macro F1, per-class metrics
  - Hierarchical consistency

Usage:
    python test_multiclass_ablation.py --checkpoint checkpoints/ablation_multiclass_B/best.pt --variant B
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.amp import autocast
from tqdm import tqdm

from models.model import MMConfig
from train_multiclass_ablation import (
    AblationModel,
    ABLATION_VARIANTS,
)
from utils.true_dataset import (
    create_dataloaders,
    NUM_FINE_PER_COARSE,
    TOTAL_FINE_CLASSES,
)

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
    return labels.argmax(dim=-1)


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_cfg_from_checkpoint(checkpoint: dict) -> MMConfig:
    cfg_dict = checkpoint.get("cfg", {})
    if not cfg_dict and "model_config" in checkpoint: # fallback
        cfg_dict = checkpoint["model_config"]
    elif not cfg_dict:
        raise KeyError("Checkpoint does not contain 'cfg' or 'model_config'.")
    valid_fields = {f.name for f in fields(MMConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    return MMConfig(**filtered)


@torch.no_grad()
def evaluate(model, loader, device, desc="Testing"):
    model.eval()

    all_coarse_true, all_coarse_pred = [], []
    all_flat_fine_true, all_flat_fine_pred = [], []

    for batch in tqdm(loader, desc=desc):
        coarse_labels = normalize_labels(batch["label"]).to(device)
        fine_labels = batch["fine_label"].to(device)
        flat_fine_labels = batch["flat_fine_label"].to(device)

        with autocast(device_type="cuda"):
            out = model(
                claim=batch["claim"],
                text_evidence=batch["content"],
                image_evidence=batch["keyframes"],
                # No coarse_labels passed -> utilizes internally predicted coarse for routing
            )

        coarse_preds = out["coarse_logits"].argmax(dim=-1)

        # Flat fine predictions
        flat_fine_preds = []
        for i in range(coarse_labels.size(0)):
            c = coarse_preds[i].item()
            n_fine_c = NUM_FINE_PER_COARSE[c]
            fine_pred_i = out["fine_logits"][i, :n_fine_c].argmax().item()
            offset = sum(NUM_FINE_PER_COARSE[:c])
            flat_fine_preds.append(offset + fine_pred_i)

        all_coarse_true.extend(coarse_labels.cpu().tolist())
        all_coarse_pred.extend(coarse_preds.cpu().tolist())
        all_flat_fine_true.extend(flat_fine_labels.cpu().tolist())
        all_flat_fine_pred.extend(flat_fine_preds)

    # Coarse metrics
    coarse_acc = accuracy_score(all_coarse_true, all_coarse_pred)
    coarse_prec = precision_score(all_coarse_true, all_coarse_pred, average="macro", zero_division=0)
    coarse_rec = recall_score(all_coarse_true, all_coarse_pred, average="macro", zero_division=0)
    coarse_f1 = f1_score(all_coarse_true, all_coarse_pred, average="macro", zero_division=0)

    # Fine metrics
    fine_acc = accuracy_score(all_flat_fine_true, all_flat_fine_pred)
    fine_prec = precision_score(all_flat_fine_true, all_flat_fine_pred, average="macro", zero_division=0)
    fine_rec = recall_score(all_flat_fine_true, all_flat_fine_pred, average="macro", zero_division=0)
    fine_f1 = f1_score(all_flat_fine_true, all_flat_fine_pred, average="macro", zero_division=0)

    # Hierarchical consistency
    consistent = sum(1 for ct, cp in zip(all_coarse_true, all_coarse_pred) if ct == cp)
    consistency = consistent / max(len(all_coarse_true), 1)

    return {
        "coarse_acc": coarse_acc,
        "coarse_precision": coarse_prec,
        "coarse_recall": coarse_rec,
        "coarse_f1": coarse_f1,
        "fine_acc": fine_acc,
        "fine_precision": fine_prec,
        "fine_recall": fine_rec,
        "fine_f1": fine_f1,
        "consistency": consistency,
        "coarse_true": all_coarse_true,
        "coarse_pred": all_coarse_pred,
        "flat_fine_true": all_flat_fine_true,
        "flat_fine_pred": all_flat_fine_pred,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical multi-class ablation study checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        help="Ablation variant key (e.g. A, B, C, D) used for evaluation",
    )
    parser.add_argument("--data-root", type=str, default="./data/TRUE_Dataset")
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    variant_key = args.variant.upper()
    if variant_key not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant '{variant_key}'. Valid variants are: {list(ABLATION_VARIANTS.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)

    cfg = build_cfg_from_checkpoint(checkpoint)
    num_fine = tuple(
        checkpoint.get("num_fine_per_coarse", list(NUM_FINE_PER_COARSE))
    )

    ablation_cfg = ABLATION_VARIANTS[variant_key]
    model = AblationModel(cfg, ablation_cfg, num_fine_per_coarse=num_fine).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)
    print(f"\nVariant: {variant_key} ({ablation_cfg.description})")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (FP32 dtype): {model_size_mb:.2f} MB")

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        path=args.data_root,
        batch_size=args.batch_size,
        shuffle_train=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[args.split]

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Evaluating split: {args.split}")
    print(f"Samples: {len(loader.dataset)}")

    # Evaluate
    metrics = evaluate(model, loader, device, desc=f"Evaluating {args.split} for {variant_key}")

    # Print results
    print(f"\n{'=' * 70}")
    print("COARSE (Binary) Results:")
    print(f"{'=' * 70}")
    print(f"  Accuracy: {metrics['coarse_acc']:.4f}")
    print(f"  Precision (macro): {metrics['coarse_precision']:.4f}")
    print(f"  Recall (macro): {metrics['coarse_recall']:.4f}")
    print(f"  F1 (macro): {metrics['coarse_f1']:.4f}")

    coarse_report = classification_report(
        metrics["coarse_true"],
        metrics["coarse_pred"],
        target_names=[COARSE_LABELS[i] for i in range(2)],
        digits=4,
        zero_division=0,
    )
    print(coarse_report)

    print(f"\n{'=' * 70}")
    print("FINE-GRAINED (8-class) Results:")
    print(f"{'=' * 70}")
    print(f"  Accuracy: {metrics['fine_acc']:.4f}")
    print(f"  Precision (macro): {metrics['fine_precision']:.4f}")
    print(f"  Recall (macro): {metrics['fine_recall']:.4f}")
    print(f"  F1 (macro): {metrics['fine_f1']:.4f}")
    print(f"  Hierarchical Consistency: {metrics['consistency']:.4f}")

    fine_report = classification_report(
        metrics["flat_fine_true"],
        metrics["flat_fine_pred"],
        target_names=[FINE_LABELS[i] for i in range(TOTAL_FINE_CLASSES)],
        digits=4,
        zero_division=0,
    )
    print(fine_report)

    # Confusion matrix
    print("\nFine-grained Confusion Matrix:")
    cm = confusion_matrix(
        metrics["flat_fine_true"],
        metrics["flat_fine_pred"],
        labels=list(range(TOTAL_FINE_CLASSES)),
    )
    fine_names = [FINE_LABELS[i] for i in range(TOTAL_FINE_CLASSES)]
    header = "          " + " ".join(f"{n[:6]:>6}" for n in fine_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:6d}" for v in row)
        print(f"{fine_names[i]:<10}{row_str}")

    # Save log
    log_path = checkpoint_path.parent / f"evaluation_{args.split}_{checkpoint_path.stem}_multiclass_ablation_{variant_key}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluating split: {args.split}\n")
        f.write(f"Variant: {variant_key} ({ablation_cfg.description})\n")
        f.write(f"Loaded checkpoint: {checkpoint_path}\n\n")
        f.write("=== COARSE (Binary) ===\n")
        f.write(f"Accuracy: {metrics['coarse_acc']:.4f}\n")
        f.write(f"Precision (macro): {metrics['coarse_precision']:.4f}\n")
        f.write(f"Recall (macro): {metrics['coarse_recall']:.4f}\n")
        f.write(f"F1 (macro): {metrics['coarse_f1']:.4f}\n")
        f.write(coarse_report + "\n\n")
        f.write("=== FINE-GRAINED (8-class) ===\n")
        f.write(f"Accuracy: {metrics['fine_acc']:.4f}\n")
        f.write(f"Precision (macro): {metrics['fine_precision']:.4f}\n")
        f.write(f"Recall (macro): {metrics['fine_recall']:.4f}\n")
        f.write(f"F1 (macro): {metrics['fine_f1']:.4f}\n")
        f.write(f"Hierarchical Consistency: {metrics['consistency']:.4f}\n")
        f.write(fine_report + "\n")
    print(f"\nSaved evaluation log to: {log_path}")

    # Save JSON output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path),
            "variant": variant_key,
            "description": ablation_cfg.description,
            "split": args.split,
            "coarse_accuracy": float(metrics["coarse_acc"]),
            "coarse_precision": float(metrics["coarse_precision"]),
            "coarse_recall": float(metrics["coarse_recall"]),
            "coarse_f1": float(metrics["coarse_f1"]),
            "fine_accuracy": float(metrics["fine_acc"]),
            "fine_precision": float(metrics["fine_precision"]),
            "fine_recall": float(metrics["fine_recall"]),
            "fine_f1": float(metrics["fine_f1"]),
            "hierarchical_consistency": float(metrics["consistency"]),
            "n_samples": len(loader.dataset),
            "coarse_predictions": metrics["coarse_pred"],
            "coarse_true_labels": metrics["coarse_true"],
            "fine_predictions": metrics["flat_fine_pred"],
            "fine_true_labels": metrics["flat_fine_true"],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved output to: {out_path}")


if __name__ == "__main__":
    main()
