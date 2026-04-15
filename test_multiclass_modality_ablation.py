"""
Test script for multiclass modality ablation variants.

Loads a saved multiclass modality ablation checkpoint and evaluates on the test set.

Usage:
    python test_multiclass_modality_ablation.py --ablation_dir checkpoints/multiclass_modality_ablation_B
"""

import os
import json
from pathlib import Path
import argparse

import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score

from utils.true_dataset import (
    create_dataloaders,
    NUM_FINE_PER_COARSE,
    TOTAL_FINE_CLASSES,
)
from models.model import MMConfig

# Reuse the modality ablation model and utilities
from train_multiclass_modality_ablation import (
    ModalityAblationConfig,
    MulticlassModalityAblationModel,
    evaluate,
    build_hierarchical_loss,
    COARSE_LABELS,
    FINE_LABELS,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    target_dir = Path(args.ablation_dir)
    if not target_dir.exists():
        raise ValueError(f"Directory not found: {target_dir}")

    device = torch.device(
        args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    DATA_ROOT = Path("./data/TRUE_Dataset")
    print("\nLoading test dataset...")
    _, _, test_loader = create_dataloaders(
        path=str(DATA_ROOT),
        batch_size=args.batch_size,
        shuffle_train=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # In evaluation, we don't strict weight the loss (for pure metric calculation it doesn't matter too much, 
    # but evaluate() uses it to report validation/test loss)
    loss_func = build_hierarchical_loss(
        device,
        lambda_coarse=1.0,
        lambda_fine=1.0,
    )

    all_results = []

    if (target_dir / "config.json").exists() and (target_dir / "best.pt").exists():
        variant_dirs = [target_dir]
    else:
        variant_dirs = [
            d
            for d in target_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists() and (d / "best.pt").exists()
        ]
        variant_dirs.sort(key=lambda d: d.name)

    if not variant_dirs:
        print(f"No valid modality ablation variant directories found in {target_dir}")
        return

    print(f"Found {len(variant_dirs)} variants to evaluate:")

    for model_dir in variant_dirs:
        with open(model_dir / "config.json", "r") as f:
            cfg_dict = json.load(f)

        variant_key = cfg_dict.get("variant", model_dir.name.split("_")[-1])
        desc = cfg_dict.get("description", "")

        print(f"\n{'═' * 70}")
        print(f"  Evaluating {variant_key}: {desc}")
        print(f"{'═' * 70}")

        mm_cfg = MMConfig(**cfg_dict["model_config"])

        mod_flags = cfg_dict.get("modality_flags", {})
        ab_cfg = ModalityAblationConfig(
            name=model_dir.name,
            description=desc,
            use_evidence=mod_flags.get("use_evidence", True),
            use_visual=mod_flags.get("use_visual", True),
        )

        model = MulticlassModalityAblationModel(
            mm_cfg, ab_cfg, num_fine_per_coarse=NUM_FINE_PER_COARSE
        ).to(device)

        checkpoint_path = model_dir / "best.pt"
        print(f"  Loading weights from {checkpoint_path.name}...")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])

        val_fine_f1_best = checkpoint.get("best_fine_f1", "N/A")
        val_coarse_f1_best = checkpoint.get("best_coarse_f1", "N/A")

        # Evaluate
        test_metrics = evaluate(
            model, test_loader, device, loss_func, desc=f"Testing {variant_key}"
        )

        coarse_report = classification_report(
            test_metrics["coarse_true"],
            test_metrics["coarse_pred"],
            target_names=[COARSE_LABELS[i] for i in range(2)],
            digits=4,
            zero_division=0,
        )
        fine_report = classification_report(
            test_metrics["flat_fine_true"],
            test_metrics["flat_fine_pred"],
            target_names=[FINE_LABELS[i] for i in range(TOTAL_FINE_CLASSES)],
            digits=4,
            zero_division=0,
        )

        log_path = model_dir / "test_evaluation_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Variant: {variant_key} - {desc}\n")
            f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
            f.write(f"Coarse Accuracy: {test_metrics['coarse_acc']:.4f}\n")
            f.write(f"Coarse F1 (macro): {test_metrics['coarse_f1']:.4f}\n")
            f.write(f"Fine Accuracy: {test_metrics['flat_fine_acc']:.4f}\n")
            f.write(f"Fine F1 (macro): {test_metrics['flat_fine_f1']:.4f}\n\n")
            f.write("Coarse Classification Report:\n")
            f.write(coarse_report + "\n")
            f.write("Fine-grained Classification Report:\n")
            f.write(fine_report + "\n")

        print(f"\n  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Coarse Accuracy: {test_metrics['coarse_acc']:.4f}")
        print(f"  Coarse F1 (macro): {test_metrics['coarse_f1']:.4f}")
        print(f"  Fine Accuracy: {test_metrics['flat_fine_acc']:.4f}")
        print(f"  Fine F1 (macro): {test_metrics['flat_fine_f1']:.4f}")
        print("\n=== Coarse Classification Report ===")
        print(coarse_report)
        print("\n=== Fine Classification Report ===")
        print(fine_report)

        all_results.append(
            {
                "variant": variant_key,
                "description": desc,
                "val_fine_f1_best": val_fine_f1_best,
                "val_coarse_f1_best": val_coarse_f1_best,
                "test_loss": float(test_metrics["loss"]),
                "test_coarse_acc": float(test_metrics["coarse_acc"]),
                "test_coarse_f1": float(test_metrics["coarse_f1"]),
                "test_fine_acc": float(test_metrics["flat_fine_acc"]),
                "test_fine_f1": float(test_metrics["flat_fine_f1"]),
            }
        )

        del model
        torch.cuda.empty_cache()

    print(f"\n{'═' * 80}")
    print("  MULTICLASS MODALITY ABLATION EVALUATION SUMMARY")
    print(f"{'═' * 80}")
    print(f"{'Variant':<8} {'Description':<35} {'Val FF1':>7} {'Val CF1':>7} {'Test FF1':>8} {'Test CF1':>8}")
    print(f"{'─' * 8} {'─' * 35} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 8}")
    for r in all_results:
        vff1 = f"{r['val_fine_f1_best']:.4f}" if isinstance(r['val_fine_f1_best'], float) else str(r['val_fine_f1_best'])
        vcf1 = f"{r['val_coarse_f1_best']:.4f}" if isinstance(r['val_coarse_f1_best'], float) else str(r['val_coarse_f1_best'])
        print(
            f"{r['variant']:<8} {r['description']:<35} "
            f"{vff1:>7} {vcf1:>7} "
            f"{r['test_fine_f1']:>8.4f} {r['test_coarse_f1']:>8.4f}"
        )
    print(f"{'═' * 80}\n")

    summary_json_path = target_dir / "multiclass_test_evaluation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_md_path = target_dir / "multiclass_test_evaluation_summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Multiclass Modality Ablation Test Evaluation Summary\n\n")
        f.write(
            "| Variant | Description | Val Fine F1 | Val Coarse F1 | Test Fine F1 | Test Coarse F1 |\n"
        )
        f.write(
            "|---------|-------------|-------------|---------------|--------------|----------------|\n"
        )
        for r in all_results:
            vff1 = f"{r['val_fine_f1_best']:.4f}" if isinstance(r['val_fine_f1_best'], float) else str(r['val_fine_f1_best'])
            vcf1 = f"{r['val_coarse_f1_best']:.4f}" if isinstance(r['val_coarse_f1_best'], float) else str(r['val_coarse_f1_best'])
            f.write(
                f"| {r['variant']} | {r['description']} | "
                f"{vff1} | {vcf1} | "
                f"{r['test_fine_f1']:.4f} | {r['test_coarse_f1']:.4f} |\n"
            )

    print(f"✓ Summary saved to {summary_json_path}")
    print(f"✓ Summary saved to {summary_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained multiclass modality ablation variants"
    )
    parser.add_argument(
        "--ablation_dir",
        type=str,
        required=True,
        help="Path to a single variant dir or a parent directory containing variant folders",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    main(args)
