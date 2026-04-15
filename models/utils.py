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
from transformers import LongformerTokenizer, LongformerModel, LongformerTokenizerFast

from transformers import logging
import math

logging.set_verbosity_error()


def video_model(type="videomae", device="auto"):
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


def vision_model(type="clip", device="auto"):
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
        raise ValueError(f"Unknown vision model type: '{type}'. Available: {available}")

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


def text_model(model_type="roberta-base", device="auto"):
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
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


def text_model_long(pt: str = "longformer", device: str = "auto"):
    MODEL_REGISTRY = {
        "longformer": {
            "model": "allenai/longformer-base-4096",
            "model_class": LongformerModel,
            "tokenizer_class": LongformerTokenizerFast,
            "trust_remote_code": False,
        },
        "longformer-large": {
            "model": "allenai/longformer-large-4096",
            "model_class": LongformerModel,
            "tokenizer_class": LongformerTokenizerFast,
            "trust_remote_code": False,
        },
        "bge-m3": {
            "model": "BAAI/bge-m3",
            "model_class": AutoModel,
            "tokenizer_class": AutoTokenizer,
            "trust_remote_code": True,
        },
    }

    if pt not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {pt}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = MODEL_REGISTRY[pt]
    print(f"Loading {pt} model ({config['model']}) on {device}...")

    tokenizer = config["tokenizer_class"].from_pretrained(
        config["model"], trust_remote_code=config["trust_remote_code"]
    )
    model = config["model_class"].from_pretrained(
        config["model"], trust_remote_code=config["trust_remote_code"]
    )

    model.eval()

    model = model.to(device)

    return tokenizer, model
