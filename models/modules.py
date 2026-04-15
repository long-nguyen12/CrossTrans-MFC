from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def masked_mean(
    x: torch.Tensor, mask: Optional[torch.Tensor], dim: int
) -> torch.Tensor:
    """
    x: (B, L, D)
    mask: (B, L) with 1 for valid positions, 0 for pad
    """
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(x.dtype)

    if mask.shape[1] != x.shape[1]:
        raise ValueError(
            f"Mask sequence length {mask.shape[1]} doesn't match x sequence length {x.shape[1]}. "
            f"x shape: {x.shape}, mask shape: {mask.shape}"
        )

    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)  # (B, 1)
    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...],
        out_dim: int,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        dims = (in_dim,) + hidden_dims + (out_dim,)
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), act, nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Multimodal Fusion Module
# -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=k_padding_mask,
            need_weights=False,
        )
        x = self.norm1(q + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class MultimodalFusionModule(nn.Module):
    """Multimodal fusion with optional bi-directional cross-attention."""

    def __init__(
        self,
        text_in_dim: int,
        img_in_dim: int,
        d_model: int,
        n_heads: int,
        out_dim: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.text_to_model = nn.Linear(text_in_dim, d_model)
        self.img_to_model = nn.Linear(img_in_dim, d_model)

        if self.bidirectional:
            self.cross_attn_layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "t_from_i": CrossAttention(d_model, n_heads, dropout),
                            "i_from_t": CrossAttention(d_model, n_heads, dropout),
                        }
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.cross_attn_layers = nn.ModuleList(
                [CrossAttention(d_model, n_heads, dropout) for _ in range(num_layers)]
            )

        # Reserved for potential gating; not used in forward.
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())

        self.fuse_mlp = MLP(
            in_dim=2 * d_model,
            hidden_dims=(d_model,),
            out_dim=out_dim,
            dropout=dropout,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        img_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = self.text_to_model(text_tokens)  # (B, Lt, D)
        i = self.img_to_model(img_tokens)  # (B, Li, D)

        img_kpm = None if img_mask is None else (img_mask == 0)
        txt_kpm = None if text_mask is None else (text_mask == 0)

        for layer in self.cross_attn_layers:
            if self.bidirectional:
                t = layer["t_from_i"](q=t, k=i, v=i, k_padding_mask=img_kpm)
                i = layer["i_from_t"](q=i, k=t, v=t, k_padding_mask=txt_kpm)
            else:
                t = layer(q=t, k=i, v=i, k_padding_mask=img_kpm)

        t_vec = masked_mean(t, text_mask, dim=1)
        i_vec = masked_mean(i, img_mask, dim=1)
        fused = torch.cat([t_vec, i_vec], dim=-1)
        return self.fuse_mlp(fused)


class MultiHeadGatedFusion(nn.Module):
    def __init__(
        self,
        dim1: int,
        dim2: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.proj1 = nn.ModuleList(
            [nn.Linear(dim1, self.head_dim) for _ in range(num_heads)]
        )
        self.proj2 = nn.ModuleList(
            [nn.Linear(dim2, self.head_dim) for _ in range(num_heads)]
        )

        self.gates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim1 + dim2, self.head_dim), nn.Sigmoid())
                for _ in range(num_heads)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """x1: (B, dim1), x2: (B, dim2)"""
        combined_input = torch.cat([x1, x2], dim=-1)

        head_outputs = []
        for i in range(self.num_heads):
            h1 = self.proj1[i](x1)
            h2 = self.proj2[i](x2)
            gate = self.gates[i](combined_input)
            head_out = gate * h1 + (1 - gate) * h2
            head_outputs.append(head_out)

        fused = torch.cat(head_outputs, dim=-1)  # (B, out_dim)
        return self.norm(self.dropout(fused))


class TemporalPositionalEncoding(nn.Module):
    """
    Injects timing/sequence information into the visual tokens.
    """

    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
