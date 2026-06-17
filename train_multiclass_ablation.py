"""
Ablation study runner for Hierarchical CrossTransVFC.

Systematically evaluates each architectural modification independently:
  A: Baseline (without PE, CrossTrans and GatedFusion)
  B: Baseline + Positional encoding
  C: Baseline + PE + Cross transformer
  D: Baseline + PE + Cross transformer + Gated fusion (full architecture)

Usage:
    python train_multiclass_ablation.py                          # run all variants
    python train_multiclass_ablation.py --variants B C           # run specific variants
    python train_multiclass_ablation.py --epochs 10 --batch_size 16
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
from models.modules import (
    MultimodalFusionModule,
    MultiHeadGatedFusion,
    TemporalPositionalEncoding,
)

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
# Ablation model: extends HierarchicalCrossTransVFC with configurable architecture
# ──────────────────────────────────────────────────────────────────────────────


class AblationModel(HierarchicalCrossTransVFC):
    """HierarchicalCrossTransVFC with runtime-configurable ablation toggles."""

    def __init__(
        self,
        cfg: MMConfig,
        ablation: AblationConfig,
        num_fine_per_coarse=NUM_FINE_PER_COARSE,
    ):
        # Call grandparent __init__ to skip HierarchicalCrossTransVFC.__init__
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

        # Hierarchical classifier (replaces flat binary classifier)
        from models.model_multiclass import HierarchicalClassifier

        self.hierarchical_classifier = HierarchicalClassifier(
            in_dim=out_fusion_dim,
            num_coarse=2,
            num_fine_per_coarse=num_fine_per_coarse,
            dropout=cfg.cls_dropout,
        )

    def forward(
        self,
        claim,
        text_evidence,
        image_evidence=None,
        labels=None,
        coarse_labels=None,
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

        # ── Hierarchical classification ──
        cls_out = self.hierarchical_classifier(h, coarse_labels=coarse_labels)

        coarse_probs = F.softmax(cls_out["coarse_logits"], dim=-1)

        return {
            "coarse_logits": cls_out["coarse_logits"],
            "coarse_probs": coarse_probs,
            "fine_logits": cls_out["fine_logits"],
            "flat_fine_logits": cls_out["flat_fine_logits"],
            "fused_features": h,
            # Backward-compatible keys
            "logits": cls_out["coarse_logits"],
            "probs": coarse_probs,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────


def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
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

        loss_fine = torch.tensor(0.0, device=device)
        n_fine_samples = 0

        for c in range(2):
            mask_c = coarse_labels == c
            if not mask_c.any():
                continue
            n_fine_c = NUM_FINE_PER_COARSE[c]
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


def compute_class_weights(dataset, num_classes, device):
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
def evaluate(model, loader, device, loss_func, desc="Evaluating"):
    model.eval()

    all_coarse_true, all_coarse_pred = [], []
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

        # Flat fine prediction: offset by coarse group
        flat_fine_preds = []
        for i in range(coarse_labels.size(0)):
            c = coarse_preds[i].item()
            n_fine_c = NUM_FINE_PER_COARSE[c]
            fine_pred_i = out["fine_logits"][i, :n_fine_c].argmax().item()
            offset = sum(NUM_FINE_PER_COARSE[:c])
            flat_fine_preds.append(offset + fine_pred_i)

        bs = coarse_labels.size(0)
        total_loss += loss.item() * bs
        n += bs

        all_coarse_true.extend(coarse_labels.cpu().tolist())
        all_coarse_pred.extend(coarse_preds.cpu().tolist())
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

    return {
        "loss": total_loss / max(n, 1),
        "coarse_acc": coarse_acc,
        "coarse_f1": coarse_f1,
        "flat_fine_acc": flat_fine_acc,
        "flat_fine_f1": flat_fine_f1,
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
):
    model.train()
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc=f"Training {ep}/{epochs}", leave=False):
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
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate one ablation variant."""

    print(f"\n{'═' * 70}")
    print(f"  ABLATION {variant_key}: {ablation_cfg.description} (seed={seed})")
    print(f"{'═' * 70}")
    epochs = args.epochs
    lr = args.lr
    warmup_ratio = 0.06
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

    # Build ablation model
    model = AblationModel(
        cfg, ablation_cfg, num_fine_per_coarse=NUM_FINE_PER_COARSE
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    coarse_weights, coarse_counts = compute_class_weights(
        train_loader.dataset, cfg.num_classes, device
    )
    fine_weights, fine_counts = compute_fine_class_weights(train_loader.dataset, device)

    loss_func = build_hierarchical_loss(
        device,
        coarse_weights=coarse_weights,
        fine_weights=fine_weights,
        lambda_coarse=args.lambda_coarse,
        lambda_fine=args.lambda_fine,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
    )
    scaler = GradScaler("cuda")

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Run dir ──
    run_dir = results_dir / f"{ablation_cfg.name}_seed{seed}"
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
                "lambda_coarse": args.lambda_coarse,
                "lambda_fine": args.lambda_fine,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_config": cfg.__dict__,
                "num_fine_per_coarse": list(NUM_FINE_PER_COARSE),
                "coarse_labels": COARSE_LABELS,
                "fine_labels": FINE_LABELS,
            },
            f,
            indent=2,
        )

    train_log_path = run_dir / "train_log.csv"
    with open(train_log_path, "w", encoding="utf-8") as f:
        f.write(
            "Epoch,Train_Loss,Val_Loss,Val_Coarse_Acc,Val_Coarse_F1,Val_Fine_Acc,Val_Fine_F1\n"
        )

    # ── Training loop ──
    best_fine_f1, best_coarse_f1 = -1.0, -1.0
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
            f"coarse_acc={metrics['coarse_acc']:.4f} | "
            f"coarse_f1={metrics['coarse_f1']:.4f} | "
            f"fine_acc={metrics['flat_fine_acc']:.4f} | "
            f"fine_f1={metrics['flat_fine_f1']:.4f}"
        )

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},{tr_loss:.4f},{metrics['loss']:.4f},"
                f"{metrics['coarse_acc']:.4f},{metrics['coarse_f1']:.4f},"
                f"{metrics['flat_fine_acc']:.4f},{metrics['flat_fine_f1']:.4f}\n"
            )

        if metrics["flat_fine_f1"] > best_fine_f1:
            best_fine_f1 = metrics["flat_fine_f1"]
            best_coarse_f1 = metrics["coarse_f1"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "best_fine_f1": best_fine_f1,
                    "best_coarse_f1": best_coarse_f1,
                    "cfg": cfg.__dict__,
                    "num_fine_per_coarse": list(NUM_FINE_PER_COARSE),
                },
                run_dir / "best.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  [{variant_key}] Early stopping at epoch {ep}")
            break

    # ── Final test ──
    checkpoint = torch.load(
        run_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])

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

    with open(run_dir / "test_log.txt", "w", encoding="utf-8") as f:
        f.write(f"Variant: {variant_key} - {ablation_cfg.description}\n")
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Coarse Accuracy: {test_metrics['coarse_acc']:.4f}\n")
        f.write(f"Coarse F1 (macro): {test_metrics['coarse_f1']:.4f}\n")
        f.write(f"Fine Accuracy: {test_metrics['flat_fine_acc']:.4f}\n")
        f.write(f"Fine F1 (macro): {test_metrics['flat_fine_f1']:.4f}\n\n")
        f.write("Coarse Classification Report:\n")
        f.write(coarse_report + "\n")
        f.write("Fine-grained Classification Report:\n")
        f.write(fine_report + "\n")

    result = {
        "variant": variant_key,
        "name": ablation_cfg.name,
        "description": ablation_cfg.description,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_fine_f1": float(best_fine_f1),
        "best_val_coarse_f1": float(best_coarse_f1),
        "test_loss": float(test_metrics["loss"]),
        "test_coarse_acc": float(test_metrics["coarse_acc"]),
        "test_coarse_f1": float(test_metrics["coarse_f1"]),
        "test_fine_acc": float(test_metrics["flat_fine_acc"]),
        "test_fine_f1": float(test_metrics["flat_fine_f1"]),
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

    # Weighted sampler (based on coarse labels)
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

    SEEDS = [42, 128, 256, 512, 1024]
    num_runs = args.num_runs
    seeds = SEEDS[:num_runs]

    print(f"\nRunning {len(variant_keys)} ablation variant(s): {variant_keys}")
    print(f"Each variant will be trained {num_runs} time(s) with seeds: {seeds}")

    metric_keys = [
        "test_loss",
        "test_coarse_acc",
        "test_coarse_f1",
        "test_fine_acc",
        "test_fine_f1",
    ]

    all_variant_summaries = []
    for key in variant_keys:
        if key not in ABLATION_VARIANTS:
            print(f"WARNING: Unknown variant '{key}', skipping.")
            continue

        results_dir = Path("checkpoints") / f"ablation_multiclass_{key}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {results_dir}\n")

        seed_results = []
        for run_seed in seeds:
            result = run_single_ablation(
                variant_key=key,
                ablation_cfg=ABLATION_VARIANTS[key],
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                args=args,
                results_dir=results_dir,
                seed=run_seed,
            )
            seed_results.append(result)

        # Aggregate results for this variant
        summary = {
            "variant": key,
            "name": ABLATION_VARIANTS[key].name,
            "description": ABLATION_VARIANTS[key].description,
            "seeds": seeds,
            "total_params": seed_results[0]["total_params"],
            "trainable_params": seed_results[0]["trainable_params"],
            "per_seed_results": seed_results,
        }
        for mk in metric_keys:
            values = [r[mk] for r in seed_results]
            summary[f"{mk}_mean"] = float(np.mean(values))
            summary[f"{mk}_std"] = float(np.std(values))
            summary[f"{mk}_values"] = values

        all_variant_summaries.append(summary)

        # Save per-variant aggregated results
        with open(results_dir / "aggregated_results.json", "w") as f:
            json.dump(summary, f, indent=2)

    # ── Summary table (mean ± std) ──
    print(f"\n{'═' * 110}")
    print(f"  ABLATION RESULTS SUMMARY ({num_runs} runs, mean ± std)")
    print(f"{'═' * 110}")
    print(
        f"{'Variant':<8} {'Description':<45} "
        f"{'Test CoarseF1':>18} {'Test FineF1':>18} {'Test FineAcc':>18}"
    )
    print(f"{'─' * 8} {'─' * 45} {'─' * 18} {'─' * 18} {'─' * 18}")
    for s in all_variant_summaries:
        print(
            f"{s['variant']:<8} {s['description']:<45} "
            f"{s['test_coarse_f1_mean']:.4f}±{s['test_coarse_f1_std']:.4f}  "
            f"{s['test_fine_f1_mean']:.4f}±{s['test_fine_f1_std']:.4f}  "
            f"{s['test_fine_acc_mean']:.4f}±{s['test_fine_acc_std']:.4f}"
        )
    print(f"{'═' * 110}\n")

    # Save overall summary
    overall_dir = Path("checkpoints")
    with open(overall_dir / "ablation_aggregated_summary.json", "w") as f:
        json.dump(all_variant_summaries, f, indent=2)

    with open(overall_dir / "ablation_aggregated_summary.md", "w") as f:
        f.write(f"# Ablation Study Results ({num_runs} runs, mean ± std)\n\n")
        f.write(
            "| Variant | Description | Params (trainable) "
            "| Test Coarse F1 | Test Fine F1 | Test Fine Acc |\n"
        )
        f.write(
            "|---------|-------------|--------------------|"
            "----------------|--------------|---------------|\n"
        )
        for s in all_variant_summaries:
            f.write(
                f"| {s['variant']} | {s['description']} | "
                f"{s['trainable_params']:,} | "
                f"{s['test_coarse_f1_mean']:.4f}±{s['test_coarse_f1_std']:.4f} | "
                f"{s['test_fine_f1_mean']:.4f}±{s['test_fine_f1_std']:.4f} | "
                f"{s['test_fine_acc_mean']:.4f}±{s['test_fine_acc_std']:.4f} |\n"
            )

    print(f"✓ All aggregated results saved to {overall_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation study for Hierarchical CrossTransVFC"
    )
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
    parser.add_argument(
        "--lambda_coarse",
        type=float,
        default=1.0,
        help="Weight for coarse (binary) loss",
    )
    parser.add_argument(
        "--lambda_fine", type=float, default=1.0, help="Weight for fine-grained loss"
    )
    parser.add_argument("--text_model", type=str, default="roberta-base")
    parser.add_argument("--long_text_model", type=str, default="longformer")
    parser.add_argument("--image_model", type=str, default="clip")
    parser.add_argument("--video_model", type=str, default="videomae")
    parser.add_argument(
        "--num_runs", type=int, default=5, help="Number of runs with different seeds"
    )
    args = parser.parse_args()
    main(args)
