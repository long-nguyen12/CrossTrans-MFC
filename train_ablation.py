"""
Ablation study runner for CrossTransVFC.

Systematically evaluates each architectural modification independently:
  A: Baseline (2-stream unidirectional, masked_mean via AttentionPooling, with TemporalPE)
  B: Baseline + Bidirectional cross-attention on claim↔visual
  C: Baseline + Evidence↔Visual third fusion stream
  D: Baseline + Bidirectional + Evidence↔Visual (full architecture)

Usage:
    python train_ablation.py                          # run all variants
    python train_ablation.py --variants B C           # run specific variants
    python train_ablation.py --epochs 10 --batch_size 16
"""

import os
import json
import time
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import get_cosine_schedule_with_warmup

from utils.true_dataset import create_dataloaders
import utils.true_dataset as true_dataset_module
from models.model import CrossTransVFC, MMConfig
from models.modules import (
    MultimodalFusionModule,
    MultiHeadGatedFusion,
    TemporalPositionalEncoding,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# Ablation variant definitions
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AblationConfig:
    """Flags toggled by each ablation variant."""

    name: str = "A_baseline"
    description: str = "Baseline 2-stream unidirectional"
    use_positional_encoding: bool = False
    use_gated_fusion: bool = False
    use_cross_transformer: bool = False


ABLATION_VARIANTS: Dict[str, AblationConfig] = {
    "A": AblationConfig(
        name="A_baseline",
        description="Baseline: without PE, CrossTrans and GatedFusion",
    ),
    "B": AblationConfig(
        name="B_positional_encoding",
        description="Baseline + Positional encoding",
        use_positional_encoding=True,
        use_gated_fusion=False,
        use_cross_transformer=False,
    ),
    "C": AblationConfig(
        name="C_cross_transformer",
        description="Baseline + PE + Cross transformer",
        use_positional_encoding=True,
        use_cross_transformer=True,
        use_gated_fusion=False,
    ),
    "D": AblationConfig(
        name="D_gated_fusion",
        description="Baseline + PE + Cross transformer + Gated fusion",
        use_positional_encoding=True,
        use_cross_transformer=True,
        use_gated_fusion=True,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Ablation model: extends CrossTransVFC with configurable architecture
# ──────────────────────────────────────────────────────────────────────────────


class AblationModel(CrossTransVFC):
    """CrossTransVFC with runtime-configurable ablation toggles."""

    def __init__(self, cfg: MMConfig, ablation: AblationConfig):
        # Call grandparent __init__ to skip CrossTransVFC.__init__
        nn.Module.__init__(self)
        self.cfg = cfg
        self.ablation = ablation

        self.use_positional_encoding = ablation.use_positional_encoding
        self.use_gated_fusion = ablation.use_gated_fusion
        self.use_cross_transformer = ablation.use_cross_transformer

        # ── Load backbone encoders (same as baseline) ──
        self._text_processor, self._text_model = self.text_model(cfg._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(
            cfg._long_pt
        )
        if not cfg.use_video:
            self._vision_processor, self._vision_model = self.vision_model(
                cfg._vision_pt
            )
            vision_dim = self._vision_model.config.projection_dim
            self._vision_hidden_dim = vision_dim
        else:
            self._video_processor, self._video_model = self.video_model(cfg._video_pt)
            video_dim = self._video_model.config.hidden_size
            self._video_hidden_dim = video_dim
        self._apply_freeze_policy()

        text_dim = self._text_model.config.hidden_size
        long_text_dim = self._long_text_model.config.hidden_size
        vis_dim = (
            self._vision_hidden_dim if not cfg.use_video else self._video_hidden_dim
        )

        if self.use_positional_encoding:
            self.temporal_pe_vision = TemporalPositionalEncoding(d_model=vis_dim)

        self.claim_video_trans = MultimodalFusionModule(
            text_in_dim=text_dim,
            img_in_dim=vis_dim,
            d_model=cfg.mfm_d_model,
            n_heads=cfg.mfm_heads,
            out_dim=cfg.mfm_out_dim,
            dropout=cfg.mfm_dropout,
            bidirectional=ablation.use_cross_transformer,
        )

        self.claim_evidence_trans = MultimodalFusionModule(
            text_in_dim=text_dim,
            img_in_dim=long_text_dim,
            d_model=cfg.mfm_d_model,
            n_heads=cfg.mfm_heads,
            out_dim=cfg.mfm_out_dim,
            dropout=cfg.mfm_dropout,
            bidirectional=False,
        )

        out_fusion_dim = (
            cfg.fusion_hidden[-1] if len(cfg.fusion_hidden) else cfg.mfm_out_dim
        )

        if self.use_gated_fusion:
            self.fusion_mlp = MultiHeadGatedFusion(
                dim1=cfg.mfm_out_dim,
                dim2=cfg.mfm_out_dim,
                out_dim=out_fusion_dim,
                dropout=cfg.mfm_dropout,
            )
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(cfg.mfm_out_dim * 2, out_fusion_dim),
                nn.ReLU(),
                nn.Dropout(cfg.mfm_dropout),
            )

        self.dropout = nn.Dropout(cfg.cls_dropout)
        self.classifier = nn.Linear(out_fusion_dim, cfg.num_classes)

    def forward(
        self,
        claim,
        text_evidence,
        image_evidence=None,
        labels=None,
    ):
        device = next(self.parameters()).device

        # ── Claims ──
        if isinstance(claim, str):
            claims = [claim]
        else:
            claims = [str(c) for c in claim]
        batch_size = len(claims)

        # ── Text evidence ──
        evidences = []
        if not isinstance(text_evidence, list):
            text_evidence = [text_evidence]
        sep_token = self._long_text_processor.sep_token or "[SEP]"
        for ev in text_evidence:
            if isinstance(ev, (list, tuple)):
                ev_texts = [
                    str(x).strip()
                    for x in ev
                    if x is not None and str(x).strip() not in ("", "nan", "None")
                ]
                evidences.append(f" {sep_token} ".join(ev_texts) if ev_texts else "")
            else:
                ev_str = str(ev) if ev is not None else ""
                evidences.append(
                    ev_str if ev_str.strip() not in ("nan", "None") else ""
                )
        while len(evidences) < batch_size:
            evidences.append("")

        # ── Encode claims ──
        claim_encoded = self._text_processor(
            claims,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.claim_max_length,
        )
        claim_encoded = {k: v.to(device) for k, v in claim_encoded.items()}
        claim_out = self._text_model(**claim_encoded)
        claim_tokens = claim_out.last_hidden_state
        claim_mask = claim_encoded["attention_mask"]

        # ── Encode evidence ──
        text_encoded = self._long_text_processor(
            evidences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.evidence_max_length,
        )
        text_encoded = {k: v.to(device) for k, v in text_encoded.items()}
        long_out = self._long_text_model(**text_encoded)
        evidence_tokens = long_out.last_hidden_state
        evidence_mask = text_encoded["attention_mask"]

        # ── Encode images ──
        image_tokens, image_mask = self._process_image(
            image_evidence, batch_size, device
        )

        if self.use_positional_encoding:
            image_tokens = self.temporal_pe_vision(image_tokens)

        # ── Fusion streams ──
        claim_visual_fused = self.claim_video_trans(
            text_tokens=claim_tokens,
            img_tokens=image_tokens,
            text_mask=claim_mask,
            img_mask=image_mask,
        )
        claim_evidence_fused = self.claim_evidence_trans(
            text_tokens=claim_tokens,
            img_tokens=evidence_tokens,
            text_mask=claim_mask,
            img_mask=evidence_mask,
        )

        if self.use_gated_fusion:
            h = self.fusion_mlp(claim_visual_fused, claim_evidence_fused)
        else:
            h = torch.cat([claim_visual_fused, claim_evidence_fused], dim=-1)
            h = self.fusion_mlp(h)

        h = self.dropout(h)
        logits = self.classifier(h)
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities (reused from train.py)
# ──────────────────────────────────────────────────────────────────────────────


def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels.argmax(dim=-1)


def build_loss(device, class_weights=None):
    return nn.CrossEntropyLoss(weight=class_weights).to(device)


def compute_class_weights(dataset, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for sample in dataset:
        label_idx = int(sample["label"].argmax().item())
        if 0 <= label_idx < num_classes:
            counts[label_idx] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device), counts


@torch.no_grad()
def evaluate(model, loader, device, loss_func, desc="Evaluating"):
    model.eval()
    all_y, all_p = [], []
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc=desc, leave=False):
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
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        all_y.extend(labels.cpu().tolist())
        all_p.extend(preds.cpu().tolist())

    acc = accuracy_score(all_y, all_p)
    f1m = f1_score(all_y, all_p, average="macro", zero_division=0)
    return {
        "loss": total_loss / max(n, 1),
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
    grad_clip=1.0,
):
    model.train()
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc=f"Training {ep}/{epochs}", leave=False):
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


# ──────────────────────────────────────────────────────────────────────────────
# Single-variant training run
# ──────────────────────────────────────────────────────────────────────────────


def run_single_ablation(
    variant_key: str,
    ablation_cfg: AblationConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args,
    results_dir: Path,
) -> Dict[str, Any]:
    """Train and evaluate one ablation variant."""

    print(f"\n{'═' * 70}")
    print(f"  ABLATION {variant_key}: {ablation_cfg.description}")
    print(f"{'═' * 70}")

    seed = 42
    epochs = args.epochs
    lr = args.lr
    warmup_ratio = 0.06
    patience = args.patience

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

    # Build ablation model
    model = AblationModel(cfg, ablation_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    class_weights, _ = compute_class_weights(
        train_loader.dataset, cfg.num_classes, device
    )
    loss_func = build_loss(device, class_weights=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
    )
    scaler = GradScaler("cuda")

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Run dir ──
    run_dir = results_dir / ablation_cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "variant": variant_key,
                "description": ablation_cfg.description,
                "ablation_flags": {
                    "use_positional_encoding": ablation_cfg.use_positional_encoding,
                    "use_cross_transformer": ablation_cfg.use_cross_transformer,
                    "use_gated_fusion": ablation_cfg.use_gated_fusion,
                },
                "epochs": epochs,
                "lr": lr,
                "seed": seed,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_config": cfg.__dict__,
            },
            f,
            indent=2,
        )

    train_log_path = run_dir / "train_log.csv"
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Acc,Val_F1\n")

    # ── Training loop ──
    best_f1, best_acc = -1.0, -1.0
    patience_counter = 0

    for ep in range(1, epochs + 1):
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
            f"  [{variant_key}] Epoch {ep}/{epochs} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"val_acc={metrics['acc']:.4f} | "
            f"val_f1={metrics['f1_macro']:.4f}"
        )

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},{tr_loss:.4f},{metrics['loss']:.4f},"
                f"{metrics['acc']:.4f},{metrics['f1_macro']:.4f}\n"
            )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_acc = metrics["acc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "best_acc": best_acc,
                },
                run_dir / "best.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  [{variant_key}] Early stopping at epoch {ep}")
            break

    # ── Final test ──
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    label2id = {"TRUE": 0, "FALSE": 1}
    id2label = {v: k for k, v in label2id.items()}

    test_metrics = evaluate(
        model, test_loader, device, loss_func, desc=f"Testing {variant_key}"
    )
    clf_report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        target_names=[id2label[i] for i in range(len(label2id))],
        digits=4,
        zero_division=0,
    )

    with open(run_dir / "test_log.txt", "w", encoding="utf-8") as f:
        f.write(f"Variant: {variant_key} - {ablation_cfg.description}\n")
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Test Accuracy: {test_metrics['acc']:.4f}\n")
        f.write(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}\n\n")
        f.write(clf_report + "\n")

    result = {
        "variant": variant_key,
        "name": ablation_cfg.name,
        "description": ablation_cfg.description,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_f1": float(best_f1),
        "best_val_acc": float(best_acc),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
    }

    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main: orchestrate all ablation runs
# ──────────────────────────────────────────────────────────────────────────────


def main(args):
    DATA_ROOT = Path("./data/TRUE_Dataset")

    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        path=str(DATA_ROOT),
        batch_size=args.batch_size,
        shuffle_train=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    # Weighted sampler
    train_dataset = train_loader.dataset
    sample_labels = torch.tensor(
        [int(s["label"].argmax().item()) for s in train_dataset], dtype=torch.long
    )
    class_counts = torch.bincount(sample_labels, minlength=2).float().clamp_min(1.0)
    sample_weights = (class_counts.sum() / (2 * class_counts))[sample_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_loader.collate_fn,
    )

    print(
        f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
        f"Test: {len(test_loader.dataset)}"
    )

    # ── Select variants ──
    if args.variants:
        variant_keys = [v.upper() for v in args.variants]
    else:
        variant_keys = list(ABLATION_VARIANTS.keys())

    print(f"\nRunning {len(variant_keys)} ablation variant(s): {variant_keys}")

    all_results = []
    for key in variant_keys:
        if key not in ABLATION_VARIANTS:
            print(f"WARNING: Unknown variant '{key}', skipping.")
            continue

        # ── Run dir ──
        results_dir = Path("checkpoints") / f"ablation_{key}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {results_dir}\n")

        result = run_single_ablation(
            variant_key=key,
            ablation_cfg=ABLATION_VARIANTS[key],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            results_dir=results_dir,
        )
        all_results.append(result)

    # ── Summary table ──
    print(f"\n{'═' * 80}")
    print("  ABLATION RESULTS SUMMARY")
    print(f"{'═' * 80}")
    print(
        f"{'Variant':<8} {'Description':<45} {'Val F1':>7} {'Test F1':>8} {'Test Acc':>9}"
    )
    print(f"{'─' * 8} {'─' * 45} {'─' * 7} {'─' * 8} {'─' * 9}")
    for r in all_results:
        print(
            f"{r['variant']:<8} {r['description']:<45} "
            f"{r['best_val_f1']:>7.4f} {r['test_f1_macro']:>8.4f} {r['test_acc']:>9.4f}"
        )
    print(f"{'═' * 80}\n")

    # Save summary
    with open(results_dir / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save markdown table
    with open(results_dir / "ablation_summary.md", "w") as f:
        f.write("# Ablation Study Results\n\n")
        f.write(
            f"| Variant | Description | Params (trainable) | Val F1 | Test F1 | Test Acc |\n"
        )
        f.write(
            f"|---------|-------------|--------------------|--------|---------|----------|\n"
        )
        for r in all_results:
            f.write(
                f"| {r['variant']} | {r['description']} | "
                f"{r['trainable_params']:,} | "
                f"{r['best_val_f1']:.4f} | {r['test_f1_macro']:.4f} | {r['test_acc']:.4f} |\n"
            )

    print(f"✓ All results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for CrossTransVFC")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Variant keys to run (e.g., A B C D). Default: all.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--text_model", type=str, default="roberta-base")
    parser.add_argument("--long_text_model", type=str, default="longformer")
    parser.add_argument("--image_model", type=str, default="clip")
    parser.add_argument("--video_model", type=str, default="videomae")
    args = parser.parse_args()
    main(args)
