"""
Hierarchical Multi-class Classification Model for CrossTransVFC.

Extends the binary CrossTransVFC with a two-level classification head:
  - Coarse level: TRUE / FALSE (2 classes)
  - Fine level: sub-labels within each coarse class
    TRUE  -> true (0), mostly_true (1), correct_attribution (2)
    FALSE -> false (0), mostly_false (1), mixture (2), fake (3), miscaptioned (4)

Supports loading pre-trained binary CrossTransVFC weights for the backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.model import CrossTransVFC, MMConfig
from models.modules import (
    MultimodalFusionModule,
    MultiHeadGatedFusion,
    TemporalPositionalEncoding,
)
from utils.true_dataset import NUM_FINE_PER_COARSE, TOTAL_FINE_CLASSES


class HierarchicalClassifier(nn.Module):
    """
    Two-level hierarchical classifier.

    - coarse_head: predicts binary TRUE/FALSE
    - fine_heads: one sub-classifier per coarse class

    At inference, uses coarse prediction to select the appropriate fine head.
    At training, uses ground-truth coarse labels to route samples.
    """

    def __init__(
        self,
        in_dim: int,
        num_coarse: int = 2,
        num_fine_per_coarse: Tuple[int, ...] = NUM_FINE_PER_COARSE,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_coarse = num_coarse
        self.num_fine_per_coarse = num_fine_per_coarse

        self.dropout = nn.Dropout(dropout)

        # Coarse-level head (binary TRUE/FALSE)
        self.coarse_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_coarse),
        )

        # Fine-grained heads (one per coarse class)
        self.fine_heads = nn.ModuleList()
        for n_fine in num_fine_per_coarse:
            self.fine_heads.append(
                nn.Sequential(
                    nn.Linear(in_dim, in_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_dim // 2, n_fine),
                )
            )

    def forward(
        self,
        h: torch.Tensor,
        coarse_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: (B, D) fused feature vector
            coarse_labels: (B,) ground-truth coarse labels for training routing.
                           If None (inference), uses predicted coarse labels.
        Returns:
            dict with:
                coarse_logits: (B, num_coarse)
                fine_logits: (B, max_fine) — padded, per-sample fine logits
                fine_labels_mask: (B, max_fine) — valid positions mask
                flat_fine_logits: (B, total_fine) — logits across all fine classes
        """
        h = self.dropout(h)
        coarse_logits = self.coarse_head(h)  # (B, num_coarse)

        # Determine routing
        if coarse_labels is not None:
            route = coarse_labels  # (B,)
        else:
            route = coarse_logits.argmax(dim=-1)  # (B,)

        B = h.size(0)
        max_fine = max(self.num_fine_per_coarse)
        total_fine = sum(self.num_fine_per_coarse)

        fine_logits = torch.full(
            (B, max_fine), float("-inf"), device=h.device, dtype=torch.float32
        )
        flat_fine_logits = torch.full(
            (B, total_fine), float("-inf"), device=h.device, dtype=torch.float32
        )

        # Compute fine logits for each coarse group
        for c in range(self.num_coarse):
            mask_c = route == c  # (B,) bool
            if not mask_c.any():
                continue

            h_c = h[mask_c]  # (N_c, D)
            logits_c = self.fine_heads[c](
                h_c
            ).float()  # (N_c, num_fine[c]), cast to fp32
            n_fine_c = self.num_fine_per_coarse[c]

            # Fill per-group fine logits
            fine_logits[mask_c, :n_fine_c] = logits_c

            # Fill flat fine logits at the correct offset
            offset = sum(self.num_fine_per_coarse[:c])
            flat_fine_logits[mask_c, offset : offset + n_fine_c] = logits_c

        return {
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "flat_fine_logits": flat_fine_logits,
        }


class HierarchicalCrossTransVFC(CrossTransVFC):
    """
    CrossTransVFC with hierarchical classification heads.

    Reuses the full backbone (text/vision encoders, cross-attention, fusion)
    from CrossTransVFC, and replaces the flat classifier with
    HierarchicalClassifier.
    """

    def __init__(self, cfg: MMConfig, num_fine_per_coarse=NUM_FINE_PER_COARSE):
        # Initialize parent CrossTransVFC (builds full backbone + binary classifier)
        super().__init__(cfg)

        # Determine fusion output dimension
        out_fusion_dim = (
            cfg.fusion_hidden[-1] if len(cfg.fusion_hidden) else cfg.mfm_out_dim
        )

        # Replace the binary classifier with hierarchical classifier
        del self.classifier
        self.hierarchical_classifier = HierarchicalClassifier(
            in_dim=out_fusion_dim,
            num_coarse=2,
            num_fine_per_coarse=num_fine_per_coarse,
            dropout=cfg.cls_dropout,
        )

    def load_pretrained_binary(self, checkpoint_path: str, device: torch.device = None):
        """
        Load pre-trained binary CrossTransVFC weights.

        Ignores `classifier.*` keys (which don't exist in hierarchical model)
        and loads everything else (backbone + fusion).
        """
        if device is None:
            device = next(self.parameters()).device

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Filter out the binary classifier weights
        filtered = {
            k: v for k, v in state_dict.items() if not k.startswith("classifier.")
        }

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"Loaded pre-trained binary weights from {checkpoint_path}")
        print(f"  Missing keys (expected - new heads): {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        return missing, unexpected

    def forward(
        self,
        claim,
        text_evidence,
        video_evidence=None,
        image_evidence=None,
        labels=None,
        coarse_labels=None,
    ):
        """
        Forward pass with hierarchical classification.

        Args:
            claim, text_evidence, video_evidence, image_evidence: same as CrossTransVFC
            labels: ignored (kept for API compatibility)
            coarse_labels: (B,) ground-truth binary labels for training routing

        Returns:
            dict with coarse_logits, fine_logits, flat_fine_logits, fused_features
        """
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
            # Backward-compatible keys
            "logits": cls_out["coarse_logits"],
            "probs": coarse_probs,
        }
