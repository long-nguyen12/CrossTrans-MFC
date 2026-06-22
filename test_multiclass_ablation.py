"""
Evaluate trained hierarchical multi-class ablation study checkpoints for CrossTransVFC.

Re-constructs the specific ablation variant Architectures and reports metrics at two levels:
  - Coarse: binary TRUE/FALSE accuracy, F1, precision, recall
  - Fine-grained: 8-class accuracy, macro F1, per-class metrics
  - Hierarchical consistency

Iterates through all ablation variants (or customized list) and generates a summary table.

Usage:
    python test_multiclass_ablation.py
    python test_multiclass_ablation.py --checkpoint-dir checkpoints
    python test_multiclass_ablation.py --variants B C
"""

import argparse
import csv
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
    if not cfg_dict and "model_config" in checkpoint:  # fallback
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
    detailed_results = []

    for batch in tqdm(loader, desc=desc, leave=False):
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
        coarse_probs = torch.softmax(out["coarse_logits"], dim=-1)

        # Flat fine predictions
        flat_fine_preds = []
        for i in range(coarse_labels.size(0)):
            c = coarse_preds[i].item()
            n_fine_c = NUM_FINE_PER_COARSE[c]
            fine_logits_i = out["fine_logits"][i, :n_fine_c]
            fine_probs_i = torch.softmax(fine_logits_i, dim=-1)
            fine_pred_i = fine_logits_i.argmax().item()
            offset = sum(NUM_FINE_PER_COARSE[:c])
            flat_fine_preds.append(offset + fine_pred_i)

            coarse_true_i = coarse_labels[i].item()
            coarse_pred_i = c
            flat_fine_true_i = flat_fine_labels[i].item()
            flat_fine_pred_i = offset + fine_pred_i
            detailed_results.append(
                {
                    "index": len(detailed_results),
                    "claim_id": batch["claim_id"][i],
                    "claim": batch["claim"][i],
                    "rating": batch["rating"][i],
                    "url": batch["url"][i],
                    "coarse_true": coarse_true_i,
                    "coarse_true_name": COARSE_LABELS[coarse_true_i],
                    "coarse_pred": coarse_pred_i,
                    "coarse_pred_name": COARSE_LABELS[coarse_pred_i],
                    "coarse_confidence": float(coarse_probs[i, coarse_pred_i].item()),
                    "coarse_correct": coarse_true_i == coarse_pred_i,
                    "fine_true": flat_fine_true_i,
                    "fine_true_name": FINE_LABELS[flat_fine_true_i],
                    "fine_pred": flat_fine_pred_i,
                    "fine_pred_name": FINE_LABELS[flat_fine_pred_i],
                    "fine_confidence": float(fine_probs_i[fine_pred_i].item()),
                    "fine_correct": flat_fine_true_i == flat_fine_pred_i,
                }
            )

        all_coarse_true.extend(coarse_labels.cpu().tolist())
        all_coarse_pred.extend(coarse_preds.cpu().tolist())
        all_flat_fine_true.extend(flat_fine_labels.cpu().tolist())
        all_flat_fine_pred.extend(flat_fine_preds)

    # Coarse metrics
    coarse_acc = accuracy_score(all_coarse_true, all_coarse_pred)
    coarse_prec = precision_score(
        all_coarse_true, all_coarse_pred, average="macro", zero_division=0
    )
    coarse_rec = recall_score(
        all_coarse_true, all_coarse_pred, average="macro", zero_division=0
    )
    coarse_f1 = f1_score(
        all_coarse_true, all_coarse_pred, average="macro", zero_division=0
    )

    # Fine metrics
    fine_acc = accuracy_score(all_flat_fine_true, all_flat_fine_pred)
    fine_prec = precision_score(
        all_flat_fine_true, all_flat_fine_pred, average="macro", zero_division=0
    )
    fine_rec = recall_score(
        all_flat_fine_true, all_flat_fine_pred, average="macro", zero_division=0
    )
    fine_f1 = f1_score(
        all_flat_fine_true, all_flat_fine_pred, average="macro", zero_division=0
    )

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
        "detailed_results": detailed_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical multi-class ablation study checkpoints."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Path to the directory containing ablation checkpoints",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="List of variants to evaluate (e.g. A B C). Default: all variants in ABLATION_VARIANTS.",
    )
    parser.add_argument("--data-root", type=str, default="./data/TRUE_Dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    if args.variants:
        variant_keys = [v.upper() for v in args.variants]
    else:
        variant_keys = list(ABLATION_VARIANTS.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data once
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        path=args.data_root,
        batch_size=args.batch_size,
        shuffle_train=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[args.split]

    print(f"Evaluating split: {args.split} ({len(loader.dataset)} samples)")
    print(f"Evaluating variants: {variant_keys}\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for key in variant_keys:
        if key not in ABLATION_VARIANTS:
            print(f"WARNING: Unknown variant '{key}', skipping.")
            continue

        ablation_cfg = ABLATION_VARIANTS[key]
        print(f"\n{'═' * 70}")
        print(f"  Evaluating Variant {key}: {ablation_cfg.description}")
        print(f"{'═' * 70}")

        base_variant_dir = Path(args.checkpoint_dir) / f"ablation_multiclass_{key}"
        if not base_variant_dir.exists():
            print(f"  [ERROR] Variant directory not found at {base_variant_dir}")
            continue

        subfolders = [d for d in base_variant_dir.iterdir() if d.is_dir()]
        if not subfolders:
            print(f"  [ERROR] No subfolders found in {base_variant_dir}")
            continue

        # Dynamically use the only subfolder present
        checkpoint_path = subfolders[0] / "best.pt"

        if not checkpoint_path.exists():
            print(f"  [ERROR] Checkpoint not found at {checkpoint_path}")
            continue

        print(f"  Loading {checkpoint_path}...")
        checkpoint = load_checkpoint(checkpoint_path, device)
        cfg = build_cfg_from_checkpoint(checkpoint)
        num_fine = tuple(
            checkpoint.get("num_fine_per_coarse", list(NUM_FINE_PER_COARSE))
        )

        model = AblationModel(cfg, ablation_cfg, num_fine_per_coarse=num_fine).to(
            device
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        metrics = evaluate(model, loader, device, desc=f"Evaluating {key}")

        print(f"  --- Coarse Metrics ---")
        print(f"  Accuracy:  {metrics['coarse_acc']:.4f}")
        print(f"  Precision: {metrics['coarse_precision']:.4f}")
        print(f"  Recall:    {metrics['coarse_recall']:.4f}")
        print(f"  F1:        {metrics['coarse_f1']:.4f}")
        print(f"  --- Fine Metrics ---")
        print(f"  Accuracy:  {metrics['fine_acc']:.4f}")
        print(f"  Precision: {metrics['fine_precision']:.4f}")
        print(f"  Recall:    {metrics['fine_recall']:.4f}")
        print(f"  F1:        {metrics['fine_f1']:.4f}")

        # Confusion matrix
        print("\n  Fine-grained Confusion Matrix:")
        cm = confusion_matrix(
            metrics["flat_fine_true"],
            metrics["flat_fine_pred"],
            labels=list(range(TOTAL_FINE_CLASSES)),
        )
        fine_names = [FINE_LABELS[i] for i in range(TOTAL_FINE_CLASSES)]
        header = "            " + " ".join(f"{n[:6]:>6}" for n in fine_names)
        print(header)
        for i, row in enumerate(cm):
            row_str = " ".join(f"{v:6d}" for v in row)
            print(f"  {fine_names[i]:<10}{row_str}")

        result_dict = {
            "variant": key,
            "description": ablation_cfg.description,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "coarse_accuracy": float(metrics["coarse_acc"]),
            "coarse_precision": float(metrics["coarse_precision"]),
            "coarse_recall": float(metrics["coarse_recall"]),
            "coarse_f1": float(metrics["coarse_f1"]),
            "fine_accuracy": float(metrics["fine_acc"]),
            "fine_precision": float(metrics["fine_precision"]),
            "fine_recall": float(metrics["fine_recall"]),
            "fine_f1": float(metrics["fine_f1"]),
            "hierarchical_consistency": float(metrics["consistency"]),
        }
        all_results.append(result_dict)

        # Save per-variant full report
        variant_log_path = (
            out_dir / f"evaluation_{args.split}_multiclass_ablation_{key}.txt"
        )
        with open(variant_log_path, "w", encoding="utf-8") as f:
            f.write(f"Evaluating split: {args.split}\n")
            f.write(f"Variant: {key} ({ablation_cfg.description})\n")
            f.write(f"Loaded checkpoint: {checkpoint_path}\n\n")
            f.write("=== COARSE (Binary) ===\n")
            f.write(f"Accuracy: {metrics['coarse_acc']:.4f}\n")
            f.write(f"Precision (macro): {metrics['coarse_precision']:.4f}\n")
            f.write(f"Recall (macro): {metrics['coarse_recall']:.4f}\n")
            f.write(f"F1 (macro): {metrics['coarse_f1']:.4f}\n\n")
            f.write("=== FINE-GRAINED (8-class) ===\n")
            f.write(f"Accuracy: {metrics['fine_acc']:.4f}\n")
            f.write(f"Precision (macro): {metrics['fine_precision']:.4f}\n")
            f.write(f"Recall (macro): {metrics['fine_recall']:.4f}\n")
            f.write(f"F1 (macro): {metrics['fine_f1']:.4f}\n")
            f.write(f"Hierarchical Consistency: {metrics['consistency']:.4f}\n\n")

            f.write("=== COARSE Classification Report ===\n")
            f.write(
                classification_report(
                    metrics["coarse_true"],
                    metrics["coarse_pred"],
                    target_names=[COARSE_LABELS[i] for i in range(2)],
                    digits=4,
                    zero_division=0,
                )
                + "\n"
            )

            f.write("=== FINE-GRAINED Classification Report ===\n")
            f.write(
                classification_report(
                    metrics["flat_fine_true"],
                    metrics["flat_fine_pred"],
                    target_names=[FINE_LABELS[i] for i in range(TOTAL_FINE_CLASSES)],
                    digits=4,
                    zero_division=0,
                )
                + "\n"
            )

        # Save per-record detailed results for this variant
        detail_json_path = out_dir / f"evaluation_{args.split}_multiclass_ablation_{key}_details.json"
        with open(detail_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics["detailed_results"], f, indent=2, ensure_ascii=False)
        
        detail_csv_path = out_dir / f"evaluation_{args.split}_multiclass_ablation_{key}_details.csv"
        with open(detail_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "claim_id",
                    "claim",
                    "rating",
                    "url",
                    "coarse_true",
                    "coarse_true_name",
                    "coarse_pred",
                    "coarse_pred_name",
                    "coarse_confidence",
                    "coarse_correct",
                    "fine_true",
                    "fine_true_name",
                    "fine_pred",
                    "fine_pred_name",
                    "fine_confidence",
                    "fine_correct",
                ],
            )
            writer.writeheader()
            writer.writerows(metrics["detailed_results"])

        del model
        torch.cuda.empty_cache()

    if not all_results:
        print("\nNo variants were evaluated (checkpoints might be missing).")
        return

    # ── Summary table ──
    print(f"\n{'═' * 130}")
    print(f"  MULTICLASS ABLATION EVALUATION SUMMARY ({args.split.upper()} SPLIT)")
    print(f"{'═' * 130}")
    print(
        f"{'Variant':<8} {'Description':<45} "
        f"{'C-Acc':>7} {'C-Prec':>7} {'C-Rec':>7} {'C-F1':>7} | "
        f"{'F-Acc':>7} {'F-Prec':>7} {'F-Rec':>7} {'F-F1':>7}"
    )
    print(
        f"{'─' * 8} {'─' * 45} "
        f"{'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}   "
        f"{'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}"
    )
    for r in all_results:
        print(
            f"{r['variant']:<8} {r['description']:<45} "
            f"{r['coarse_accuracy']:>7.4f} {r['coarse_precision']:>7.4f} "
            f"{r['coarse_recall']:>7.4f} {r['coarse_f1']:>7.4f} | "
            f"{r['fine_accuracy']:>7.4f} {r['fine_precision']:>7.4f} "
            f"{r['fine_recall']:>7.4f} {r['fine_f1']:>7.4f}"
        )
    print(f"{'═' * 130}\n")

    # Save summary
    summary_json_path = out_dir / f"ablation_evaluation_summary_{args.split}.json"
    with open(summary_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_md_path = out_dir / f"ablation_evaluation_summary_{args.split}.md"
    with open(summary_md_path, "w") as f:
        f.write(
            f"# Multiclass Ablation Evaluation Results ({args.split.title()} Split)\n\n"
        )
        f.write(
            "| Variant | Description | Params | Coarse Acc | Coarse Prec | Coarse Rec | Coarse F1 | Fine Acc | Fine Prec | Fine Rec | Fine F1 | Consistency |\n"
        )
        f.write(
            "|---------|-------------|--------|------------|-------------|------------|-----------|----------|-----------|----------|---------|-------------|\n"
        )
        for r in all_results:
            f.write(
                f"| {r['variant']} | {r['description']} | "
                f"{r['trainable_params']:,} | "
                f"{r['coarse_accuracy']:.4f} | {r['coarse_precision']:.4f} | "
                f"{r['coarse_recall']:.4f} | {r['coarse_f1']:.4f} | "
                f"{r['fine_accuracy']:.4f} | {r['fine_precision']:.4f} | "
                f"{r['fine_recall']:.4f} | {r['fine_f1']:.4f} | "
                f"{r['hierarchical_consistency']:.4f} |\n"
            )

    print(f"✓ Output saved to {out_dir}")


if __name__ == "__main__":
    main()
