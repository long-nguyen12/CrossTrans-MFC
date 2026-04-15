import os
import json
from pathlib import Path
import argparse

import torch
from sklearn.metrics import classification_report, precision_score, recall_score

from utils.true_dataset import create_dataloaders
from models.model import MMConfig

# We reuse the ablation model classes and utilities from train_ablation
from train_ablation import AblationConfig, AblationModel, evaluate, build_loss

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
    # create_dataloaders returns train, val, test. We just need test.
    _, _, test_loader = create_dataloaders(
        path=str(DATA_ROOT),
        batch_size=args.batch_size,
        shuffle_train=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Test samples: {len(test_loader.dataset)}\n")

    label2id = {"TRUE": 0, "FALSE": 1}
    id2label = {v: k for k, v in label2id.items()}
    loss_func = build_loss(device)

    all_results = []

    # Check if target_dir is a specific variant dir or a parent dir of variants
    if (target_dir / "config.json").exists() and (target_dir / "best.pt").exists():
        variant_dirs = [target_dir]
    else:
        # Find all ablation variant subdirectories
        variant_dirs = [
            d
            for d in target_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists() and (d / "best.pt").exists()
        ]
        variant_dirs.sort(key=lambda d: d.name)

    if not variant_dirs:
        print(f"No valid ablation variant directories found in {target_dir}")
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

        ab_flags = cfg_dict.get("ablation_flags", {})
        ab_cfg = AblationConfig(
            name=model_dir.name,
            description=desc,
            use_positional_encoding=ab_flags.get("use_positional_encoding", False),
            use_gated_fusion=ab_flags.get("use_gated_fusion", False),
            use_cross_transformer=ab_flags.get("use_cross_transformer", False),
        )

        model = AblationModel(mm_cfg, ab_cfg).to(device)

        checkpoint_path = model_dir / "best.pt"
        print(f"  Loading weights from {checkpoint_path.name}...")

        # Load with weights_only=False to allow numpy scalars
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])

        val_f1_best = checkpoint.get("best_f1", cfg_dict.get("best_val_f1", "N/A"))

        # Evaluate
        test_metrics = evaluate(
            model, test_loader, device, loss_func, desc=f"Testing {variant_key}"
        )

        test_precision = precision_score(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            average="macro",
            zero_division=0,
        )
        test_recall = recall_score(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            average="macro",
            zero_division=0,
        )

        clf_report = classification_report(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            target_names=[id2label[i] for i in range(len(label2id))],
            digits=4,
            zero_division=0,
        )

        log_path = model_dir / "test_evaluation_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Variant: {variant_key} - {desc}\n")
            f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
            f.write(f"Test Accuracy: {test_metrics['acc']:.4f}\n")
            f.write(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(clf_report + "\n")

        print(f"\n  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {test_metrics['acc']:.4f}")
        print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        print("\n" + clf_report)

        all_results.append(
            {
                "variant": variant_key,
                "description": desc,
                "val_f1_best": val_f1_best,
                "test_precision": float(test_precision),
                "test_recall": float(test_recall),
                "test_f1_macro": float(test_metrics["f1_macro"]),
                "test_acc": float(test_metrics["acc"]),
            }
        )

        del model
        torch.cuda.empty_cache()

    # ── Summary table ──
    print(f"\n{'═' * 80}")
    print("  ABLATION EVALUATION SUMMARY")
    print(f"{'═' * 80}")
    print(
        f"{'Variant':<8} {'Description':<40} {'Val F1':>7} {'Test Prec':>9} {'Test Rec':>8} {'Test F1':>8} {'Test Acc':>9}"
    )
    print(f"{'─' * 8} {'─' * 40} {'─' * 7} {'─' * 9} {'─' * 8} {'─' * 8} {'─' * 9}")
    for r in all_results:
        vf1 = (
            f"{r['val_f1_best']:.4f}"
            if isinstance(r["val_f1_best"], float)
            else str(r["val_f1_best"])
        )
        print(
            f"{r['variant']:<8} {r['description']:<40} "
            f"{vf1:>7} {r['test_precision']:>9.4f} {r['test_recall']:>8.4f} {r['test_f1_macro']:>8.4f} {r['test_acc']:>9.4f}"
        )
    print(f"{'═' * 80}\n")

    summary_json_path = target_dir / "test_evaluation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_md_path = target_dir / "test_evaluation_summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# Ablation Test Evaluation Summary\n\n")
        f.write(
            "| Variant | Description                            | Val F1 | Test Prec | Test Rec | Test F1 | Test Acc |\n"
        )
        f.write(
            "|---------|----------------------------------------|--------|-----------|----------|---------|----------|\n"
        )
        for r in all_results:
            vf1 = (
                f"{r['val_f1_best']:.4f}"
                if isinstance(r["val_f1_best"], float)
                else str(r["val_f1_best"])
            )
            f.write(
                f"| {r['variant']} | {r['description']} | "
                f"{vf1} | {r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_f1_macro']:.4f} | {r['test_acc']:.4f} |\n"
            )

    print(f"✓ Summary saved to {summary_json_path}")
    print(f"✓ Summary saved to {summary_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained ablation variants")
    parser.add_argument(
        "--ablation_dir",
        type=str,
        required=True,
        help="Path to a single variant dir or a master directory containing variant folders (e.g. checkpoints/ablation_A)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()
    main(args)
