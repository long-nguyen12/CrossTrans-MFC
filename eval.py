import os
from pathlib import Path

import numpy as np

import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.read_data import create_dataloaders
from utils import FocalLoss

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------
# Main
# -----------------------------
def main():
    DATA_ROOT = Path("data/mocheg")

    # ====== Training hyperparams ======
    seed = 42
    batch_size = 32
    num_workers = 8

    # Model config
    from models.model import (
        CrossTransVFC,
        MMConfig,
    )

    cfg = MMConfig(
        num_classes=3,
        freeze_bert=True,
        freeze_clip=True,
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
    print(f"Using device: {device}")

    # Load data
    print("\nLoading datasets...")
    # train_loader, val_loader, test_loader = create_dataloaders(
    #     path=str(DATA_ROOT),
    #     batch_size=batch_size,
    #     shuffle_train=True,
    #     num_workers=num_workers,
    #     pin_memory=True if device.type == "cuda" else False,
    # )

    # print(f"Train batches: {len(train_loader)}")
    # print(f"Val batches: {len(val_loader)}")
    # print(f"Test batches: {len(test_loader)}")

    # Create label mapping for checkpoint saving
    label2id = {"supported": 0, "NEI": 1, "refuted": 2}
    id2label = {v: k for k, v in label2id.items()}

    # Create model
    model = CrossTransVFC(cfg)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    best_checkpoint = "./checkpoints/best_1/best.pt"
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    print(checkpoint.keys())
    print(checkpoint["cfg"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return
    # Test
    y_true, y_pred = [], []
    model.eval()
    total_test_loss = 0.0
    n_test = 0

    loss_func = FocalLoss(gamma=2.0, alpha=0.25)
    loss_func = loss_func.to(device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            labels = batch["label"].argmax(dim=-1)
            labels = labels.to(device)

            out = model(
                claim=batch["claim"],
                text_evidence=batch["text_evidence"],
                image_evidence=batch["image_evidence"],
                labels=labels,
            )

            # Get raw logits for loss computation
            logits = out["logits"]

            # Compute loss externally
            loss = loss_func(logits, labels)

            # Get predictions for metrics
            pred = logits.argmax(dim=-1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(labels.cpu().tolist())

            total_test_loss += loss.item() * labels.size(0)
            n_test += labels.size(0)

    # Calculate test metrics
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    test_loss = total_test_loss / n_test

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 (macro): {test_f1:.4f}")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(len(label2id))],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
