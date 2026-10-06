"""
Encoder abstraction for DNA and Protein branches.

Each encoder takes tokenized input and returns (embeddings, mask):
  - embeddings: (B, L_compressed, d_model) — sequence embeddings
  - mask:       (B, L_compressed)          — True=valid positions

Encoders used by the shipped checkpoint:
  - CNNEncoder:     one-hot DNA → Conv1D residual blocks (DNA branch)
  - ESM2Encoder:    real ESM2 forward pass (used once, to embed the protein)
  - ESM2EmbEncoder: holds the trained projection for those embeddings
"""

import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Base encoder interface
# ═══════════════════════════════════════════════════════════════════

class BaseEncoder(nn.Module):
    """Abstract encoder. Subclasses must implement forward() and output_dim."""

    def __init__(self):
        super().__init__()
        self._output_dim = None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: tokenized input, shape varies by encoder type
            mask:   (B, L_raw) bool, True=real tokens
        Returns:
            embeddings: (B, L_compressed, d_out)
            mask:       (B, L_compressed) bool
        """
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
# CNN Encoder (baseline)
# ═══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """1D Conv → BatchNorm → ReLU, with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = (stride == 1 and in_channels == out_channels)
        if not self.use_residual:
            self.residual = nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        residual = x if self.use_residual else self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + residual


class CNNEncoder(BaseEncoder):
    """
    One-hot → residual Conv1D blocks → projection to d_model.

    Input:  (B, vocab_size, L_raw) one-hot float32
    Output: (B, L_compressed, d_model) embeddings
    """

    def __init__(self, vocab_size, conv_blocks, d_model, dropout=0.1):
        super().__init__()
        layers = []
        curr_ch = vocab_size
        total_stride = 1
        for blk in conv_blocks:
            layers.append(ConvBlock(curr_ch, blk["out_channels"], blk["kernel"],
                                    blk["stride"], dropout))
            curr_ch = blk["out_channels"]
            total_stride *= blk["stride"]

        self.conv_stack = nn.Sequential(*layers)
        self.proj = nn.Conv1d(curr_ch, d_model, 1, bias=False) if curr_ch != d_model else nn.Identity()
        self._output_dim = d_model
        self._total_stride = total_stride

    def forward(self, tokens):
        """
        Args:
            tokens: (B, vocab_size, L_raw) — one-hot encoded sequences
        Returns:
            embeddings: (B, L_compressed, d_model)
            mask:       (B, L_compressed) — all True (no padding info in one-hot)
        """
        # Create mask from input (all non-zero positions are real)
        mask_raw = (tokens.sum(dim=1) > 0).float()  # (B, L_raw)
        B, _, L_raw = tokens.shape

        out = self.conv_stack(tokens)               # (B, curr_ch, L_compressed)
        out = self.proj(out)                        # (B, d_model, L_compressed)
        out = out.transpose(1, 2)                   # (B, L_compressed, d_model)

        # Downsample mask
        L_comp = out.size(1)
        if L_comp > 0:
            mask = F.interpolate(mask_raw.unsqueeze(1), size=L_comp, mode='nearest') \
                     .squeeze(1).bool()
            if not mask.any(dim=1).all():
                empty_rows = ~mask.any(dim=1)
                mask[empty_rows, 0] = True
        else:
            mask = torch.ones(B, 1, dtype=torch.bool, device=out.device)

        return out, mask


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# ESM2 Encoder — real pretrained model via HuggingFace transformers
# ═══════════════════════════════════════════════════════════════════

class ESM2Encoder(BaseEncoder):
    """
    ESM2 pretrained protein encoder via HuggingFace transformers.

    Downloads model on first use (requires internet), caches locally.

    Input:  (B, L_aa) int64 — ESM2 token IDs (from ESM2Tokenizer)
    Output: (B, L_aa, d_model) embeddings

    ESM2 models:
      esm2_t6_8M_UR50D   —  8M params, 320-dim, ~50MB download
      esm2_t12_35M_UR50D — 35M params, 480-dim, ~140MB
      esm2_t30_150M_UR50D — 150M params, 640-dim, ~600MB
    """

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D",
                 d_model=128, freeze=True, cache_dir=None,
                 local_files_only=False):
        super().__init__()
        self.model_name = model_name

        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

        # Load ESM2
        self.esm = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.esm_dim = self.esm.config.hidden_size  # 320 for t6, 480 for t12, etc.

        # Project ESM2 output → d_model
        self.proj = nn.Linear(self.esm_dim, d_model) if self.esm_dim != d_model else nn.Identity()
        self._output_dim = d_model

        if freeze:
            for p in self.esm.parameters():
                p.requires_grad = False
            self.esm.eval()  # always eval mode for frozen

        n_params = sum(p.numel() for p in self.esm.parameters())
        n_trainable = sum(p.numel() for p in self.esm.parameters() if p.requires_grad)
        print(f"ESM2Encoder: {model_name} | dim={self.esm_dim}→{d_model} | "
              f"freeze={freeze} | params={n_params/1e6:.0f}M ({n_trainable/1e6:.1f}M trainable)")

    def forward(self, tokens):
        """
        Args:
            tokens: (B, L_prot) int64 — ESM2 token IDs (0=pad, 1=<cls>, 2=<eos>, 4-23=AA)
        Returns:
            embeddings: (B, L_prot, d_model)
            mask:       (B, L_prot) bool — True=real token
        """
        mask = (tokens != 0)  # 0 = padding

        # ESM2 forward
        with torch.set_grad_enabled(not all(p.requires_grad for p in self.esm.parameters())):
            esm_out = self.esm(input_ids=tokens, attention_mask=mask)
            emb = esm_out.last_hidden_state  # (B, L, esm_dim)

        # NaN guard
        if torch.isnan(emb).any():
            emb = torch.nan_to_num(emb, nan=0.0)

        # Project to d_model
        emb = self.proj(emb)

        return emb, mask


# ═══════════════════════════════════════════════════════════════════
# ESM2 Embedding Encoder — holds the trained projection layer
# ═══════════════════════════════════════════════════════════════════

class ESM2EmbEncoder(BaseEncoder):
    """
    Passthrough encoder for pre-computed ESM2 embeddings.

    Inference runs ESM2 itself (see ``extract_esm2_embedding_realtime`` in
    inference.py) and feeds the projected hidden states straight into the
    model, so this encoder carries no backbone — it exists to own the trained
    ``proj`` weight and to build the padding mask.

    ``esm_dim`` (640 for esm2_t30) is read from the checkpoint's own projection
    shape by ``load_model()``.

    Input:  (B, L, esm_dim) float — ESM2 hidden states, already projected
    Output: (B, L, d_model) float32
    """

    def __init__(self, d_model: int = 128, esm_dim: int = None):
        super().__init__()
        if esm_dim is None:
            raise ValueError("esm_dim is required (read it from the checkpoint's proj shape)")
        self.esm_dim = esm_dim
        self.model_name = "real-time"

        self.proj = nn.Linear(self.esm_dim, d_model) if self.esm_dim != d_model else nn.Identity()
        self._output_dim = d_model

        print(f"ESM2EmbEncoder: dim={self.esm_dim}→{d_model}")

    def forward(self, tokens):
        """
        Args:
            tokens: (B, L, esm_dim) float — ESM2 embeddings
        Returns:
            embeddings: (B, L, d_model)
            mask:       (B, L) bool — True where the embedding is non-zero
        """
        # Mask: positions where the embedding has non-zero energy
        mask = (tokens.abs().sum(dim=-1) > 1e-8)
        # Safety: ensure at least 1 position is valid per sample (prevent 0-length attention)
        if not mask.any(dim=1).all():
            # For any sample with all-False mask, set first position to True
            empty_rows = ~mask.any(dim=1)
            mask[empty_rows, 0] = True
        emb = tokens.float()
        emb = self.proj(emb)
        return emb, mask


# ═══════════════════════════════════════════════════════════════════
# Encoder factory
# ═══════════════════════════════════════════════════════════════════

DNA_CONV_BLOCKS = [
    {"kernel": 7,  "stride": 1, "out_channels": 64},
    {"kernel": 11, "stride": 1, "out_channels": 128},    # full resolution
    {"kernel": 15, "stride": 1, "out_channels": 128},
]

PROTEIN_CONV_BLOCKS = [
    {"kernel": 5,  "stride": 1, "out_channels": 64},
    {"kernel": 7,  "stride": 1, "out_channels": 128},    # full resolution
    {"kernel": 9,  "stride": 1, "out_channels": 128},
]


# Inference builds its encoders directly:
#     CNNEncoder(config.DNA_VOCAB, DNA_CONV_BLOCKS, d_model, dropout)
#     ESM2EmbEncoder(d_model=d_model, esm_dim=<from checkpoint>)
# see ``load_model()`` in inference.py.
