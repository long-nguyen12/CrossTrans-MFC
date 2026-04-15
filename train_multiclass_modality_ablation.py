"""
Multiclass modality ablation study runner for HierarchicalCrossTransVFC.

Systematically evaluates the contribution of each modality by zeroing out
specific inputs while keeping the full hierarchical architecture intact:
  B: No visual  (claim + evidence text only)
  C: No evidence (claim + visual only)

Usage:
    python train_multiclass_modality_ablation.py                          # run all variants
    python train_multiclass_modality_ablation.py --variants B             # run specific variant
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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

# ──────────────────────────────────────────────────────────────────────────────
# Modality ablation variant definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModalityAblationConfig:
    """Flags toggled by each modality ablation variant."""

    name: str = "full_model"
    description: str = "Full model (all modalities)"
    use_evidence: bool = True
    use_visual: bool = True


MODALITY_ABLATION_VARIANTS: Dict[str, ModalityAblationConfig] = {
    "B": ModalityAblationConfig(
        name="B_no_visual",
        description="No visual (claim + evidence text only)",
        use_evidence=True,
        use_visual=False,
    ),
    "C": ModalityAblationConfig(
        name="C_no_evidence",
        description="No evidence text (claim + visual only)",
        use_evidence=False,
        use_visual=True,
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# Modality ablation model: extends HierarchicalCrossTransVFC with modality zeroing
# ──────────────────────────────────────────────────────────────────────────────

class MulticlassModalityAblationModel(HierarchicalCrossTransVFC):
    """HierarchicalCrossTransVFC with runtime-configurable modality zeroing."""

    def __init__(
        self,
        cfg: MMConfig,
        ablation: ModalityAblationConfig,
        num_fine_per_coarse=NUM_FINE_PER_COARSE,
    ):
        super().__init__(cfg, num_fine_per_coarse=num_fine_per_coarse)
        self.ablation = ablation
        self.use_evidence = ablation.use_evidence
        self.use_visual = ablation.use_visual

    def forward(
        self,
        claim,
        text_evidence,
        image_evidence=None,
        labels=None,
        coarse_labels=None,
    ):
        device = next(self.parameters()).device

        # ---- Reuse CrossTransVFC encoding pipeline ----
        if isinstance(claim, str):
            claims = [claim]
        else:
            claims = [str(c) for c in claim]
        batch_size = len(claims)

        # Encode text evidence
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

        # Encode claims
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

        # Encode evidence (long text)
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

        # Encode images
        image_tokens, image_mask = self._process_image(
            image_evidence, batch_size, device
        )

        # ════════════════════════════════════════════════════════════════
        # MODALITY ZEROING: zero out disabled modality tokens and masks
        # ════════════════════════════════════════════════════════════════
        if not self.use_evidence:
            evidence_tokens = torch.zeros_like(evidence_tokens)
            evidence_mask = torch.zeros_like(evidence_mask)

        if not self.use_visual:
            image_tokens = torch.zeros_like(image_tokens)
            image_mask = torch.zeros_like(image_mask)

        # Temporal PE
        image_tokens = self.temporal_pe_vision(image_tokens)

        # Cross-attention fusion
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

        # Gated fusion
        h = self.fusion_mlp(claim_visual_fused, claim_evidence_fused)
        h = self.dropout(h)

        # Hierarchical classification
        cls_out = self.hierarchical_classifier(h, coarse_labels=coarse_labels)

        # Also compute coarse probabilities for compatibility
        coarse_probs = F.softmax(cls_out["coarse_logits"], dim=-1)

        return {
            "coarse_logits": cls_out["coarse_logits"],
            "coarse_probs": coarse_probs,
            "fine_logits": cls_out["fine_logits"],
            "flat_fine_logits": cls_out["flat_fine_logits"],
            "fused_features": h,
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
    coarse_f1 = f1_score(all_coarse_true, all_coarse_pred, average="macro", zero_division=0)
    flat_fine_acc = accuracy_score(all_flat_fine_true, all_flat_fine_pred)
    flat_fine_f1 = f1_score(all_flat_fine_true, all_flat_fine_pred, average="macro", zero_division=0)

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
    ablation_cfg: ModalityAblationConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args,
    results_dir: Path,
) -> Dict[str, Any]:
    print(f"\n{'═' * 70}")
    print(f"  MODALITY ABLATION {variant_key}: {ablation_cfg.description}")
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

    model = MulticlassModalityAblationModel(
        cfg, ablation_cfg, num_fine_per_coarse=NUM_FINE_PER_COARSE
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  use_evidence={ablation_cfg.use_evidence}, use_visual={ablation_cfg.use_visual}")

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

    run_dir = results_dir / ablation_cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "variant": variant_key,
                "description": ablation_cfg.description,
                "modality_flags": {
                    "use_evidence": ablation_cfg.use_evidence,
                    "use_visual": ablation_cfg.use_visual,
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
        f.write("Epoch,Train_Loss,Val_Loss,Val_Coarse_Acc,Val_Coarse_F1,Val_Fine_Acc,Val_Fine_F1\n")

    best_fine_f1, best_coarse_f1 = -1.0, -1.0
    patience_counter = 0

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            ep, epochs, device, loss_func, grad_clip=1.0,
        )
        metrics = evaluate(model, val_loader, device, loss_func)

        print(
            f"  [{variant_key}] Epoch {ep}/{epochs} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"c_acc={metrics['coarse_acc']:.4f} | "
            f"c_f1={metrics['coarse_f1']:.4f} | "
            f"f_acc={metrics['flat_fine_acc']:.4f} | "
            f"f_f1={metrics['flat_fine_f1']:.4f}"
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
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
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
# Main: orchestrate all modality ablation runs
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

    if args.variants:
        variant_keys = [v.upper() for v in args.variants]
    else:
        variant_keys = list(MODALITY_ABLATION_VARIANTS.keys())

    print(f"\nRunning {len(variant_keys)} modality ablation variant(s): {variant_keys}")

    all_results = []
    results_dir = None
    for key in variant_keys:
        if key not in MODALITY_ABLATION_VARIANTS:
            print(f"WARNING: Unknown variant '{key}', skipping.")
            continue

        results_dir = Path("checkpoints") / f"multiclass_modality_ablation_{key}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {results_dir}\n")

        result = run_single_ablation(
            variant_key=key,
            ablation_cfg=MODALITY_ABLATION_VARIANTS[key],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            results_dir=results_dir,
        )
        all_results.append(result)

    print(f"\n{'═' * 80}")
    print("  MULTICLASS MODALITY ABLATION RESULTS SUMMARY")
    print(f"{'═' * 80}")
    print(
        f"{'Variant':<8} {'Description':<35} {'Val FF1':>7} {'Val CF1':>7} {'Test FF1':>8} {'Test CF1':>8}"
    )
    print(f"{'─' * 8} {'─' * 35} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 8}")
    for r in all_results:
        print(
            f"{r['variant']:<8} {r['description']:<35} "
            f"{r['best_val_fine_f1']:>7.4f} {r['best_val_coarse_f1']:>7.4f} "
            f"{r['test_fine_f1']:>8.4f} {r['test_coarse_f1']:>8.4f}"
        )
    print(f"{'═' * 80}\n")

    summary_dir = Path("checkpoints") / "multiclass_modality_ablation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "multiclass_modality_ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(summary_dir / "multiclass_modality_ablation_summary.md", "w") as f:
        f.write("# Multiclass Modality Ablation Study Results\n\n")
        f.write(
            f"| Variant | Description | Params (trainable) | Val Fine F1 | Val Coarse F1 | Test Fine F1 | Test Coarse F1 |\n"
        )
        f.write(
            f"|---------|-------------|--------------------|-------------|---------------|--------------|----------------|\n"
        )
        for r in all_results:
            f.write(
                f"| {r['variant']} | {r['description']} | "
                f"{r['trainable_params']:,} | "
                f"{r['best_val_fine_f1']:.4f} | {r['best_val_coarse_f1']:.4f} | "
                f"{r['test_fine_f1']:.4f} | {r['test_coarse_f1']:.4f} |\n"
            )

    print(f"✓ All results saved to {summary_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiclass modality ablation study for HierarchicalCrossTransVFC"
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Variant keys to run (e.g., B C). Default: all.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lambda_coarse", type=float, default=1.0)
    parser.add_argument("--lambda_fine", type=float, default=1.0)
    parser.add_argument("--text_model", type=str, default="roberta-base")
    parser.add_argument("--long_text_model", type=str, default="longformer")
    parser.add_argument("--image_model", type=str, default="clip")
    parser.add_argument("--video_model", type=str, default="videomae")
    args = parser.parse_args()
    main(args)
