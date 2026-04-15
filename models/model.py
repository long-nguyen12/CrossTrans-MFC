import argparse
from dataclasses import dataclass
from typing import Tuple
import os
import sys
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import AutoTokenizer, AutoModel, AutoProcessor
from transformers import LongformerTokenizer, LongformerModel

from transformers import logging
import math

logging.set_verbosity_error()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.vid_extractor import extract_long_video
from utils.frame_extractor import _extract_image_features
from models.modules import (
    MultimodalFusionModule,
    MultiHeadGatedFusion,
    TemporalPositionalEncoding,
)


@dataclass
class MMConfig:
    _claim_pt: str = "roberta-base"
    _long_pt: str = "longformer"
    _vision_pt: str = "clip"
    _video_pt: str = "videomae"  # Options: "videomae", "vjepa2", "clip_keyframe"

    freeze_text: bool = True
    freeze_long_text: bool = True
    freeze_vision: bool = True
    freeze_video: bool = True
    use_video: bool = False
    unfreeze_text_last_n: int = 0
    unfreeze_long_last_n: int = 0

    claim_max_length: int = 128
    evidence_max_length: int = 512

    mfm_d_model: int = 256
    mfm_heads: int = 8
    mfm_out_dim: int = 256
    mfm_dropout: float = 0.1

    fusion_hidden: Tuple[int, ...] = (256,)
    cls_dropout: float = 0.4
    num_classes: int = 3


class CrossTransVFC(nn.Module):
    def __init__(self, cfg: MMConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders
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
        self.temporal_pe_vision = TemporalPositionalEncoding(d_model=vis_dim)

        self.claim_video_trans = MultimodalFusionModule(
            text_in_dim=text_dim,
            img_in_dim=vis_dim,
            d_model=cfg.mfm_d_model,
            n_heads=cfg.mfm_heads,
            out_dim=cfg.mfm_out_dim,
            dropout=cfg.mfm_dropout,
            bidirectional=True,
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

        fusion_in = 2 * cfg.mfm_out_dim
        out_fusion_dim = cfg.fusion_hidden[-1] if len(cfg.fusion_hidden) else fusion_in

        self.fusion_mlp = MultiHeadGatedFusion(
            dim1=cfg.mfm_out_dim,
            dim2=cfg.mfm_out_dim,
            out_dim=out_fusion_dim,
            dropout=cfg.mfm_dropout,
        )

        self.dropout = nn.Dropout(cfg.cls_dropout)
        self.classifier = nn.Linear(out_fusion_dim, cfg.num_classes)

    def video_model(self, type="videomae", device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {type} model on {device}...")

        model_map = {
            "videomae": {
                "id": "MCG-NJU/videomae-base",
                "needs_torch_load": False,
                "requires_trc": True,
                "processor_type": "auto",
            },
            "vjepa2": {
                "id": "facebook/vjepa2-vitl-fpc64-256",
                "needs_torch_load": True,
                "requires_trc": True,
                "processor_type": "auto",
            },
        }

        if type not in model_map:
            raise ValueError(
                f"Unknown video model type: {type}. Available: {list(model_map.keys())}"
            )

        args = model_map[type]
        model_id = args["id"]
        trust_remote_code = args["requires_trc"]

        try:
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=False,
            )
            model.to(device)
            return processor, model
        except Exception as e:
            raise RuntimeError(f"Failed to load {type} model ({model_id}): {e}")

    def vision_model(self, type="clip", device="auto"):
        model_map = {
            "clip": {
                "id": "openai/clip-vit-base-patch32",
                "requires_trc": False,
            },
            "siglip": {
                "id": "google/siglip-so400m-patch14-384",
                "requires_trc": False,
            },
            "dinov2": {
                "id": "facebook/dinov2-base",
                "requires_trc": True,
            },
            "vit": {
                "id": "google/vit-base-patch16-224",
                "requires_trc": False,
            },
        }

        if type not in model_map:
            available = list(model_map.keys())
            raise ValueError(
                f"Unknown vision model type: '{type}'. Available: {available}"
            )

        config = model_map[type]
        model_id = config["id"]
        trust_remote = config.get("requires_trc", False)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {type} vision model ({model_id}) on {device}...")

        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=trust_remote,
            )

            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                trust_remote_code=trust_remote,
            )

            model.to(device)
            return processor, model
        except Exception as e:
            error_msg = f"Failed to load {type} model ({model_id}): {e}"
            print(error_msg)
            raise RuntimeError(error_msg)

    def text_model(self, model_type="roberta-base", device="auto"):
        MODEL_REGISTRY = {
            "roberta-base": "roberta-base",
            "roberta-large": "roberta-large",
            "bert-base-uncased": "bert-base-uncased",
            "distilbert-base-uncased": "distilbert-base-uncased",
            "allenai/scibert_scivocab_uncased": "allenai/scibert_scivocab_uncased",
            "clip": "open",
        }

        if model_type in MODEL_REGISTRY:
            model_id = MODEL_REGISTRY[model_type]
        else:
            model_id = model_type

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"Invalid model_id: {model_id}")

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        print(f"Loading {model_id} on {device}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=False,
                use_fast=True,
            )

            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                local_files_only=os.getenv("HF_LOCAL_ONLY", "false").lower() == "true",
            )

            model.to(device)
            return tokenizer, model
        except Exception as e:
            error_msg = f"Failed to load text model ({model_id}): {e}"
            warnings.warn(error_msg, RuntimeWarning)
            raise RuntimeError(error_msg) from e

    def text_model_long(self, pt="longformer"):
        if pt == "longformer":
            processor = LongformerTokenizer.from_pretrained(
                "allenai/longformer-base-4096"
            )
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        else:
            raise ValueError(f"Unknown long text model: {pt}")
        return processor, model

    def _set_module_trainable(self, module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad = trainable

    def _unfreeze_last_n_transformer_layers(self, module: nn.Module, n: int) -> None:
        if n <= 0:
            return
        encoder = getattr(module, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            return
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def _apply_freeze_policy(self) -> None:
        self._set_module_trainable(self._text_model, not self.cfg.freeze_text)
        self._set_module_trainable(self._long_text_model, not self.cfg.freeze_long_text)
        if not self.cfg.use_video:
            self._set_module_trainable(self._vision_model, not self.cfg.freeze_vision)
        else:
            self._set_module_trainable(self._video_model, not self.cfg.freeze_video)

        self._unfreeze_last_n_transformer_layers(
            self._text_model, self.cfg.unfreeze_text_last_n
        )
        self._unfreeze_last_n_transformer_layers(
            self._long_text_model, self.cfg.unfreeze_long_last_n
        )

    def _process_videos(self, video_evidence, batch_size, device):
        if not isinstance(video_evidence, list):
            video_evidence = [video_evidence]

        if len(video_evidence) != batch_size:
            print(
                f"video_evidence length {len(video_evidence)} does not match batch_size {batch_size}; padding with zeros."
            )

        video_hidden_dim = self._video_hidden_dim
        all_video_features = []
        all_video_masks = []

        for vid in video_evidence:
            if vid is None or not isinstance(vid, str) or not os.path.exists(vid):
                video_feat = torch.zeros(1, video_hidden_dim, device=device)  # [1, D]
                video_mask = torch.zeros(1, device=device)  # [1]
            else:
                try:
                    print(f"Extracting features from video: {vid}")
                    video_feat = extract_long_video(
                        self._video_model, self._video_processor, vid
                    )  # Returns [1, T, D]
                    video_feat = video_feat.squeeze(0).to(device)
                    video_mask = torch.ones(video_feat.shape[0], device=device)  # [T]
                except Exception as e:
                    print(f"Error processing video {vid}: {e}")
                    video_feat = torch.zeros(
                        1, video_hidden_dim, device=device
                    )  # [1, D]
                    video_mask = torch.zeros(1, device=device)  # [1]

            all_video_features.append(video_feat)  # [T, D]
            all_video_masks.append(video_mask)  # [T]

        while len(all_video_features) < batch_size:
            all_video_features.append(torch.zeros(1, video_hidden_dim, device=device))
            all_video_masks.append(torch.zeros(1, device=device))

        all_video_features = all_video_features[:batch_size]
        all_video_masks = all_video_masks[:batch_size]

        max_clips = max(feat.shape[0] for feat in all_video_features)

        padded_features = []
        padded_masks = []

        for feat, mask in zip(all_video_features, all_video_masks):
            feat = feat.to(device)
            mask = mask.to(device)
            current_len = feat.shape[0]
            if current_len < max_clips:
                pad_len = max_clips - current_len
                feat = torch.cat(
                    [feat, torch.zeros(pad_len, video_hidden_dim, device=device)], dim=0
                )
                mask = torch.cat([mask, torch.zeros(pad_len, device=device)], dim=0)

            padded_features.append(feat)
            padded_masks.append(mask)

        video_tokens = torch.stack(padded_features, dim=0)  # [B, T, D]
        video_mask = torch.stack(padded_masks, dim=0)  # [B, T]

        return video_tokens, video_mask

    def _process_image(self, image_evidence, batch_size, device):
        if not isinstance(image_evidence, list):
            image_evidence = [image_evidence]

        if len(image_evidence) != batch_size:
            print(
                f"image_evidence length {len(image_evidence)} does not match batch_size {batch_size}; padding with zeros."
            )

        image_hidden_dim = self._vision_hidden_dim
        all_image_features = []
        all_image_masks = []

        for img_item in image_evidence:
            if not img_item:
                all_image_features.append(
                    torch.zeros(1, image_hidden_dim, device=device)
                )
                all_image_masks.append(torch.zeros(1, device=device))
                continue

            try:
                if isinstance(img_item, list):
                    item_feats = []
                    for single_img in img_item:
                        if isinstance(single_img, str) and os.path.exists(single_img):
                            try:
                                single_img = Image.open(single_img).convert("RGB")
                            except Exception as e:
                                print(f"Error opening image {single_img}: {e}")
                                continue

                        f = _extract_image_features(
                            self._vision_model,
                            self._vision_processor,
                            single_img,
                            device,
                        )
                        if f is not None:
                            if f.dim() == 1:
                                f = f.unsqueeze(0)
                            elif f.dim() == 3:
                                f = f.squeeze(0)
                            item_feats.append(f)

                    if item_feats:
                        image_feat = torch.cat(item_feats, dim=0)  # [T, D]
                    else:
                        raise ValueError(
                            f"All image extractions failed for item {img_item}"
                        )
                else:
                    if isinstance(img_item, str) and os.path.exists(img_item):
                        _img = Image.open(img_item).convert("RGB")
                    else:
                        _img = img_item

                    image_feat = _extract_image_features(
                        self._vision_model,
                        self._vision_processor,
                        _img,
                        device,
                    )
                    if image_feat is None:
                        raise ValueError(f"Feature extraction failed for {img_item}")

                    # Normalize dimensions to [T, D]
                    if image_feat.dim() == 1:
                        image_feat = image_feat.unsqueeze(0)
                    elif image_feat.dim() == 3:
                        image_feat = image_feat.squeeze(0)

                image_feat = image_feat.to(device)
                image_mask = torch.ones(image_feat.shape[0], device=device)  # [T]
            except Exception:
                image_feat = torch.zeros(1, image_hidden_dim, device=device)  # [1, D]
                image_mask = torch.zeros(1, device=device)  # [1]

            all_image_features.append(image_feat)  # [T, D]
            all_image_masks.append(image_mask)  # [T]

        while len(all_image_features) < batch_size:
            all_image_features.append(torch.zeros(1, image_hidden_dim, device=device))
            all_image_masks.append(torch.zeros(1, device=device))

        all_image_features = all_image_features[:batch_size]
        all_image_masks = all_image_masks[:batch_size]

        max_clips = max(feat.shape[0] for feat in all_image_features)

        padded_features = []
        padded_masks = []

        for feat, mask in zip(all_image_features, all_image_masks):
            feat = feat.to(device)
            mask = mask.to(device)
            current_len = feat.shape[0]
            if current_len < max_clips:
                pad_len = max_clips - current_len
                feat = torch.cat(
                    [feat, torch.zeros(pad_len, image_hidden_dim, device=device)], dim=0
                )
                mask = torch.cat([mask, torch.zeros(pad_len, device=device)], dim=0)

            padded_features.append(feat)
            padded_masks.append(mask)

        image_tokens = torch.stack(padded_features, dim=0)  # [B, T, D]
        image_mask = torch.stack(padded_masks, dim=0)  # [B, T]

        return image_tokens, image_mask

    def forward(
        self,
        claim,
        text_evidence,
        video_evidence=None,
        image_evidence=None,
        labels=None,
    ):
        device = next(self.parameters()).device

        # ---- Process claims ----
        if isinstance(claim, str):
            claims = [claim]
        else:
            claims = [str(c) for c in claim]

        batch_size = len(claims)

        # ---- Process text evidence ----
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

        # ---- Encode claims (short text) ----
        claim_encoded = self._text_processor(
            claims,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.claim_max_length,
        )
        claim_encoded = {k: v.to(device) for k, v in claim_encoded.items()}
        claim_out = self._text_model(**claim_encoded)
        claim_tokens = claim_out.last_hidden_state  # (B, Lc, Dt)
        claim_mask = claim_encoded["attention_mask"]  # (B, Lc)

        # ---- Encode evidence (long text) ----
        text_encoded = self._long_text_processor(
            evidences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.evidence_max_length,
        )
        text_encoded = {k: v.to(device) for k, v in text_encoded.items()}
        long_out = self._long_text_model(**text_encoded)
        evidence_tokens = long_out.last_hidden_state  # (B, Le, Dl)
        evidence_mask = text_encoded["attention_mask"]  # (B, Le)

        ### TODO: Add video processing

        # ---- Encode images ----
        image_tokens, image_mask = self._process_image(
            image_evidence, batch_size, device
        )  # (B, Ti, Di), (B, Ti)

        # Add temporal positional encoding
        image_tokens = self.temporal_pe_vision(image_tokens)

        claim_visual_fused = self.claim_video_trans(
            text_tokens=claim_tokens,
            img_tokens=image_tokens,
            text_mask=claim_mask,
            img_mask=image_mask,
        )  # (B, D_mfm)

        claim_evidence_fused = self.claim_evidence_trans(
            text_tokens=claim_tokens,
            img_tokens=evidence_tokens,
            text_mask=claim_mask,
            img_mask=evidence_mask,
        )  # (B, D_mfm)

        h = self.fusion_mlp(claim_visual_fused, claim_evidence_fused)
        h = self.dropout(h)
        logits = self.classifier(h)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)

        out = {"logits": logits, "probs": probs}
        return out


@torch.no_grad()
def run_quick_test(device_str: str = ""):
    device = torch.device(
        device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = MMConfig(num_classes=2)
    model = CrossTransVFC(cfg).to(device)
    model.eval()
    print(model)
    import glob

    image_evidence = glob.glob("data\\TRUE_Dataset\\test_output\\10156802\\*")

    out = model(
        claim=["This claim is for smoke testing."],
        text_evidence=["This is a short evidence sentence."],
        image_evidence=[image_evidence],
    )

    logits = out["logits"]
    probs = out["probs"]
    preds = logits.argmax(dim=-1)

    print(f"Device: {device}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Probs shape: {tuple(probs.shape)}")
    print(f"Predictions: {preds.cpu().tolist()}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Quick smoke test for model forward.")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_quick_test(device_str=args.device)
