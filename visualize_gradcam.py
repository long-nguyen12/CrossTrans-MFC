"""Explain a CrossTransVFC prediction with CLIP Grad-CAM and cross-attention.

Cross-attention sees one pooled embedding per frame, so its maps are
claim-token x frame maps. Spatial Grad-CAM uses CLIP patch activations and
the final classifier logit, including both the attention and pooling paths.
"""

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.model import CrossTransVFC, MMConfig
from models.model_multiclass import HierarchicalCrossTransVFC
from models.modules import MultimodalFusionModule
from utils.true_dataset import NUM_FINE_PER_COARSE, clean_data, resolve_keyframe_path


@torch.enable_grad()
def explain(model, claim, text_evidence, images, head="coarse", class_index=None):
    """Explain one sample. Fine indices are flat indices in the predicted group.

    Returns raw maps; no gradient is accumulated in model parameters.
    Only the final image/claim cross-attention layer is captured.
    """
    if model.cfg.use_video or model.cfg._vision_pt != "clip":
        raise ValueError("Spatial Grad-CAM currently requires the CLIP image backbone.")
    if not images:
        raise ValueError("At least one image/keyframe is required.")
    if head not in ("coarse", "fine"):
        raise ValueError("head must be 'coarse' or 'fine'.")

    layer = model.claim_video_trans.cross_attn_layers[-1]
    directions = (
        {"t_from_i": layer["t_from_i"], "i_from_t": layer["i_from_t"]}
        if model.claim_video_trans.bidirectional else {"t_from_i": layer}
    )
    previous_capture = {name: module.capture_attention for name, module in directions.items()}
    previous_image_grad = getattr(model, "image_grad_enabled", False)
    was_training = model.training
    activations = []

    def capture_patches(module, inputs, output):
        activations.append(output)

    # Patches still influence CLS before the final CLIP self-attention block.
    target_layer = model._vision_model.vision_model.encoder.layers[-1].layer_norm1
    hook = target_layer.register_forward_hook(capture_patches)
    try:
        model.eval()
        model.image_grad_enabled = True
        for module in directions.values():
            module.capture_attention = True

        output = model(claim=[claim], text_evidence=[text_evidence], image_evidence=[images])
        if head == "fine" and "flat_fine_logits" not in output:
            raise ValueError("The fine head requires a hierarchical checkpoint.")
        logits = output["flat_fine_logits" if head == "fine" else "logits"][0]
        predicted_index = int(logits.argmax())
        target_index = predicted_index if class_index is None else class_index
        if not 0 <= target_index < logits.numel():
            raise ValueError(f"Class index must be between 0 and {logits.numel() - 1}.")
        if not torch.isfinite(logits[target_index]):
            raise ValueError("Choose a fine class within the predicted coarse group.")
        if len(activations) != len(images):
            raise RuntimeError("Not every input frame produced CLIP patch activations.")

        weights = [module.attention_weights for module in directions.values()]
        gradients = torch.autograd.grad(logits[target_index], activations + weights)
        spatial_maps = []
        for activation, gradient in zip(activations, gradients[: len(activations)]):
            patches, patch_gradients = activation[:, 1:], gradient[:, 1:]
            channel_weights = patch_gradients.mean(dim=1, keepdim=True)
            cam = (channel_weights * patches).sum(dim=-1).clamp_min(0)[0]
            side = math.isqrt(cam.numel())
            if side * side != cam.numel():
                raise RuntimeError("CLIP patches must form a square grid.")
            spatial_maps.append(cam.reshape(side, side).detach().cpu().numpy())

        attention = {}
        attention_gradients = gradients[len(activations) :]
        for name, weight, gradient in zip(directions, weights, attention_gradients):
            raw = weight.mean(dim=1)[0]
            weighted = (weight * gradient).clamp_min(0).mean(dim=1)[0]
            if name == "i_from_t":
                raw, weighted = raw.T, weighted.T
            attention[name] = {
                "raw": raw.detach().cpu().numpy(),
                "gradient_weighted": weighted.detach().cpu().numpy(),
            }

        encoded = model._text_processor(
            claim, truncation=True, max_length=model.cfg.claim_max_length,
        )
        tokens = model._text_processor.convert_ids_to_tokens(encoded["input_ids"])
        return {
            "head": head,
            "target_index": target_index,
            "predicted_index": predicted_index,
            "target_logit": float(logits[target_index].detach()),
            "target_probability": float(logits.softmax(dim=-1)[target_index].detach()),
            "tokens": tokens,
            "spatial_maps": np.stack(spatial_maps),
            "attention": attention,
        }
    finally:
        hook.remove()
        model.image_grad_enabled = previous_image_grad
        for name, module in directions.items():
            module.capture_attention = previous_capture[name]
            module.attention_weights = None
        model.train(was_training)


def save_visualizations(result, images, image_paths, processor, output_dir):
    """Overlay CAM on the exact resized/cropped input seen by CLIP."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_processor = processor.image_processor
    label = result.get("target_label", str(result["target_index"]))
    title = f"{result['head']} class {label} | p={result['target_probability']:.3f}"
    arrays = {"spatial_gradcam": result["spatial_maps"]}
    for index, (image, cam) in enumerate(zip(images, result["spatial_maps"])):
        pixels = processor(images=image, return_tensors="pt")["pixel_values"][0]
        mean = torch.tensor(image_processor.image_mean)[:, None, None]
        std = torch.tensor(image_processor.image_std)[:, None, None]
        display = (pixels.cpu() * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
        normalized = cam / cam.max() if cam.max() > 0 else np.zeros_like(cam)
        heatmap = F.interpolate(
            torch.from_numpy(normalized)[None, None], size=display.shape[:2],
            mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title(f"Frame {index}: original")
        axes[1].imshow(display)
        axes[1].set_title("CLIP input crop")
        axes[2].imshow(display)
        overlay = axes[2].imshow(
            heatmap, cmap="jet", vmin=0, vmax=1, alpha=0.5 * heatmap,
        )
        axes[2].set_title("Grad-CAM" if cam.max() > 0 else "Grad-CAM: no positive attribution")
        for axis in axes:
            axis.axis("off")
        fig.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)
        fig.suptitle(title)
        fig.savefig(output_dir / f"frame_{index:02d}_gradcam.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    token_labels = [f"{i}: {token.replace(chr(288), ' ')}" for i, token in enumerate(result["tokens"])]
    for direction, maps in result["attention"].items():
        fig, axes = plt.subplots(
            1, 2, figsize=(max(10, len(images) * 0.8), max(5, len(token_labels) * 0.22)),
        )
        for axis, (kind, data) in zip(axes, maps.items()):
            arrays[f"{direction}_{kind}"] = data
            plotted = axis.imshow(data, aspect="auto", cmap="magma", vmin=0, vmax=data.max() or 1)
            axis.set_xticks(range(len(images)), labels=[str(i) for i in range(len(images))])
            axis.set_yticks(range(len(token_labels)), labels=token_labels, fontsize=8)
            axis.set_xlabel("Frame index")
            axis.set_ylabel("Claim token index")
            axis.set_title("Raw attention" if kind == "raw" else "Positive attention x logit gradient")
            fig.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
        caption = "Claim queries -> frame keys" if direction == "t_from_i" else "Frame queries -> claim keys (transposed)"
        fig.suptitle(f"Last cross-attention layer: {caption}\n{title}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{direction}_attention.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    np.savez_compressed(output_dir / "attributions.npz", **arrays)
    summary = {key: value for key, value in result.items() if key not in ("spatial_maps", "attention")}
    summary["frames"] = [str(path) for path in image_paths]
    summary["spatial_positive_max"] = [float(cam.max()) for cam in result["spatial_maps"]]
    summary["spatial_normalization"] = "Each frame is normalized by its own positive maximum for display."
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-json", required=True, help="One TRUE dataset record, e.g. data/TRUE_Dataset/test/10156802.json")
    parser.add_argument("--data-root", default="data/TRUE_Dataset")
    parser.add_argument("--images", nargs="+", help="Override keyframes with these image paths, in the given order")
    parser.add_argument("--head", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--class-index", type=int, help="Default: predicted class. Fine indices are flat (0-7).")
    parser.add_argument("--output-dir", default="plots/gradcam")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.sample_json, encoding="utf-8") as file:
        sample = json.load(file)
    claim = clean_data(sample.get("claim", ""))
    if not claim:
        parser.error("The sample must contain a non-empty claim.")
    # Match test_multiclass.py: the evidence branch receives article content.
    evidence = clean_data(sample.get("content", ""))
    claim_id = str(sample.get("video_information", {}).get("video_id", ""))
    image_paths = args.images or resolve_keyframe_path(claim_id, args.data_root)
    if not image_paths:
        parser.error("No keyframes found; supply --images with local image paths.")
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg_fields = {field.name for field in fields(MMConfig)}
    cfg = MMConfig(**{key: value for key, value in checkpoint["cfg"].items() if key in cfg_fields})
    if cfg.use_video or cfg._vision_pt != "clip":
        parser.error("This visualization requires a CLIP image checkpoint.")
    state_dict = checkpoint["state_dict"]
    if any(key.startswith("hierarchical_classifier.") for key in state_dict):
        model = HierarchicalCrossTransVFC(
            cfg, num_fine_per_coarse=tuple(checkpoint.get("num_fine_per_coarse", NUM_FINE_PER_COARSE)),
        )
    else:
        model = CrossTransVFC(cfg)
    # Earlier CLIP checkpoints used one-way fusion. Restore that saved
    # architecture before strict loading rather than changing its weights.
    if "claim_video_trans.cross_attn_layers.0.attn.in_proj_weight" in state_dict:
        model.claim_video_trans = MultimodalFusionModule(
            text_in_dim=model._text_model.config.hidden_size,
            img_in_dim=model._vision_hidden_dim,
            d_model=cfg.mfm_d_model, n_heads=cfg.mfm_heads,
            out_dim=cfg.mfm_out_dim, dropout=cfg.mfm_dropout, bidirectional=False,
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    label2id = checkpoint.get("label2id", {})
    del checkpoint, state_dict
    result = explain(model, claim, evidence, images, args.head, args.class_index)
    result.update({"claim": claim, "checkpoint": args.checkpoint, "sample_json": args.sample_json})
    if args.head == "coarse":
        result["target_label"] = next(
            (label for label, index in label2id.items() if index == result["target_index"]),
            str(result["target_index"]),
        )
    save_visualizations(result, images, image_paths, model._vision_processor, args.output_dir)
    print(f"Target: {result['head']} class {result['target_index']}, p={result['target_probability']:.4f}")
    print(f"Saved {len(images)} Grad-CAM images, {len(result['attention'])} attention figures, and raw maps to {args.output_dir}")


if __name__ == "__main__":
    main()
