# CrossTrans-MFC: Hierarchical Multimedia Fact-Checking

CrossTrans-MFC is a multi-modal, deep learning framework for hierarchical multimedia fact-checking. Designed to operate on the TRUE dataset, the repository provides tools for evaluating and training models that predict both coarse-grained authenticity (Binary: `TRUE` / `FALSE`) and fine-grained categorizations (8-class: `true`, `mostly_true`, `correct_attribution`, `false`, `mostly_false`, `mixture`, `fake`, `miscaptioned`).

## Features

- **Multi-Modal Architecture**: Incorporates features from multiple modalities:
  - Text Features: RoBERTa (for short claims/text)
  - Long Text Features: Longformer (for detailed textual evidence)
  - Image Features: CLIP (for video keyframes)
  - Video Temporal Features: VideoMAE (for spatial-temporal video context)
- **Hierarchical Verification Workflow**: Jointly predicts coarse binary labels and fine-grained taxonomy via a combined Cross-Entropy loss.
- **Two-Stage Fine-Tuning Process**: Supports starting from a pre-trained binary CrossTransVFC checkpoint and subsequently fine-tuning robust hierarchical classification heads.

## Repository Structure

```text
CrossTrans-MFC/
├── models/                     
├── utils/                      
├── data/                       
├── checkpoints/                
├── train_multiclass.py         
├── test_multiclass.py          
├── train_*.py / test_*.py      
└── eval.py / test.py           
```

## Dataset Setup

The codebase expects the TRUE dataset to be organized in the `data/TRUE_Dataset` directory. 
By default, data loading processes (like `true_dataset.py`) read from this directory to construct the unified data loading pipelines used in training and evaluation.

## Usage

### 1. Training

You can train the hierarchical multi-class model from scratch or load a pre-trained binary model.

**Train from Scratch:**
```bash
python train_multiclass.py --batch_size 32 --epochs 30 --lr 3e-5
```

**Fine-tune from a Pre-trained Binary Checkpoint:**
```bash
python train_multiclass.py \
    --pretrained_ckpt checkpoints/cross_trans_vfc/best.pt \
    --batch_size 32
```

**Freeze Backbone (Train Custom Classifiers Only):**
```bash
python train_multiclass.py \
    --pretrained_ckpt checkpoints/cross_trans_vfc/best.pt \
    --freeze_backbone
```

If you are caching video features for the first time, you can utilize the internal fallback tools located in `utils/vid_extractor.py`.

### 2. Evaluation

Test a hierarchical multi-class checkpoint and export detailed predictions (accuracy, macro-F1, precision, recall) and consistency metrics.

```bash
python test_multiclass.py \
    --checkpoint checkpoints/hierarchical_multiclass/best.pt \
    --split test \
    --batch-size 8
```

This will produce hierarchical metric aggregations, fine-grained confusion matrices, and detailed JSON/CSV outputs mapping predictions against truths.
