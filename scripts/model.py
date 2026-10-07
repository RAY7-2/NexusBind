"""
NexusBind model: TF-DNA binding prediction with cross-attention interpretability.

Composes a DNA encoder (one-hot CNN), a protein encoder (frozen ESM2
embeddings), optional self-attention layers per branch, bidirectional
cross-attention, and a prediction head.

The last cross-attention layer is what makes the model interpretable:
its Protein→DNA weights localize the DNA binding site, and its DNA→Protein
weights localize the protein DNA-binding domain. 
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════════
# Transformer Encoder
# ═══════════════════════════════════════════════════════════════════

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, mask=None):
        key_pad_mask = ~mask if mask is not None else None
        return self.encoder(x, src_key_padding_mask=key_pad_mask)


# ═══════════════════════════════════════════════════════════════════
# Cross-Attention Layer
# ═══════════════════════════════════════════════════════════════════

class CrossAttentionLayer(nn.Module):
    """Bidirectional cross-attention with Pre-LN."""

    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()

        # DNA → Protein
        self.dna_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dna_ln1 = nn.LayerNorm(d_model)
        self.dna_ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.dna_ln2 = nn.LayerNorm(d_model)

        # Protein → DNA
        self.prot_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.prot_ln1 = nn.LayerNorm(d_model)
        self.prot_ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.prot_ln2 = nn.LayerNorm(d_model)

    def forward(self, dna_emb, prot_emb, dna_mask=None, prot_mask=None,
                return_weights=False):
        dna_key_mask = ~dna_mask if dna_mask is not None else None
        prot_key_mask = ~prot_mask if prot_mask is not None else None

        # DNA attends to Protein (Pre-LN)
        dna_normed = self.dna_ln1(dna_emb)
        dna_att, dna_attn_w = self.dna_cross(
            dna_normed, prot_emb, prot_emb,
            key_padding_mask=prot_key_mask,
            need_weights=return_weights, average_attn_weights=True,
        )
        dna_emb = dna_emb + dna_att
        dna_emb = dna_emb + self.dna_ff(self.dna_ln2(dna_emb))

        # Protein attends to DNA (Pre-LN)
        prot_normed = self.prot_ln1(prot_emb)
        prot_att, prot_attn_w = self.prot_cross(
            prot_normed, dna_emb, dna_emb,
            key_padding_mask=dna_key_mask,
            need_weights=return_weights, average_attn_weights=True,
        )
        prot_emb = prot_emb + prot_att
        prot_emb = prot_emb + self.prot_ff(self.prot_ln2(prot_emb))

        if return_weights:
            return dna_emb, prot_emb, {"dna_to_prot": dna_attn_w, "prot_to_dna": prot_attn_w}
        return dna_emb, prot_emb


# ═══════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════

class TFBindingPretrainedModel(nn.Module):
    """
    TF-DNA binding prediction with pluggable encoders.

    Architecture:
        DNA encoder → [Self-Attn] ─┐
                                    ├→ Cross-Attn → Pooling → MLP → score
        Protein encoder → [Self-Attn]┘
    """

    def __init__(self,
                 dna_encoder: nn.Module,
                 protein_encoder: nn.Module,
                 d_model: int = 128,
                 n_heads: int = 4,
                 d_ff: int = 256,
                 dna_self_attn_layers: int = 2,
                 protein_self_attn_layers: int = 2,
                 cross_attn_layers: int = 2,
                 mlp_hidden: list = None,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.d_model = d_model

        if mlp_hidden is None:
            mlp_hidden = [256, 128]

        # ── Encoders ──
        self.dna_encoder = dna_encoder
        self.prot_encoder = protein_encoder

        # Project encoder outputs to d_model if needed
        self.dna_proj = nn.Linear(dna_encoder.output_dim, d_model) \
            if dna_encoder.output_dim != d_model else nn.Identity()
        self.prot_proj = nn.Linear(protein_encoder.output_dim, d_model) \
            if protein_encoder.output_dim != d_model else nn.Identity()

        # ── Positional encoding ──
        self.dna_pos = PositionalEncoding(d_model)
        self.prot_pos = PositionalEncoding(d_model)

        # ── Self-attention (optional, can be 0 layers) ──
        self.dna_self_attn = TransformerEncoder(d_model, n_heads, dna_self_attn_layers, d_ff, dropout) \
            if dna_self_attn_layers > 0 else nn.Identity()
        self.prot_self_attn = TransformerEncoder(d_model, n_heads, protein_self_attn_layers, d_ff, dropout) \
            if protein_self_attn_layers > 0 else nn.Identity()

        # ── Cross-attention ──
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(cross_attn_layers)
        ])

        # ── Prediction head ──
        head_in = 4 * d_model
        layers = []
        for h in mlp_hidden:
            layers.extend([nn.Linear(head_in, h), nn.GELU(), nn.Dropout(dropout)])
            head_in = h
        layers.append(nn.Linear(head_in, 1))
        self.head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dna_tokens, prot_tokens, return_weights=False):
        """
        Args:
            dna_tokens:  encoder-specific format
            prot_tokens: encoder-specific format
            return_weights: if True, also return cross-attention weight matrices
        Returns:
            logits: (B,) or (logits, attn_weights_list)
        """
        # ── Encode ──
        dna_emb, dna_mask = self.dna_encoder(dna_tokens)
        prot_emb, prot_mask = self.prot_encoder(prot_tokens)

        # ── Project to d_model ──
        dna_emb = self.dna_proj(dna_emb)
        prot_emb = self.prot_proj(prot_emb)

        # ── Positional encoding + Self-attention ──
        dna_emb = self.dna_pos(dna_emb)
        prot_emb = self.prot_pos(prot_emb)

        if isinstance(self.dna_self_attn, TransformerEncoder):
            dna_emb = self.dna_self_attn(dna_emb, dna_mask)
        if isinstance(self.prot_self_attn, TransformerEncoder):
            prot_emb = self.prot_self_attn(prot_emb, prot_mask)

        # ── Cross-attention ──
        all_weights = []
        for layer in self.cross_layers:
            if return_weights:
                dna_emb, prot_emb, w = layer(dna_emb, prot_emb, dna_mask, prot_mask,
                                             return_weights=True)
                all_weights.append(w)
            else:
                dna_emb, prot_emb = layer(dna_emb, prot_emb, dna_mask, prot_mask)

        # ── Global pooling ──
        #   mean: set padding to 0  so it doesn't affect sum
        #   max:  set padding to -inf so it's never selected
        if dna_mask is not None:
            dna_m_mean = dna_emb.masked_fill(~dna_mask.unsqueeze(-1), 0.0)
            dna_mean = dna_m_mean.sum(dim=1) / dna_mask.sum(dim=1, keepdim=True).clamp(min=1)

            dna_m_max = dna_emb.masked_fill(~dna_mask.unsqueeze(-1), float('-inf'))
            dna_max = dna_m_max.amax(dim=1)
        else:
            dna_mean, dna_max = dna_emb.mean(dim=1), dna_emb.amax(dim=1)

        if prot_mask is not None:
            prot_m_mean = prot_emb.masked_fill(~prot_mask.unsqueeze(-1), 0.0)
            prot_mean = prot_m_mean.sum(dim=1) / prot_mask.sum(dim=1, keepdim=True).clamp(min=1)

            prot_m_max = prot_emb.masked_fill(~prot_mask.unsqueeze(-1), float('-inf'))
            prot_max = prot_m_max.amax(dim=1)
        else:
            prot_mean, prot_max = prot_emb.mean(dim=1), prot_emb.amax(dim=1)

        pooled = torch.cat([dna_mean, dna_max, prot_mean, prot_max], dim=-1)
        logits = self.head(pooled).squeeze(-1)

        if return_weights:
            return logits, all_weights
        return logits

    @torch.no_grad()
    def predict_binding_sites(self, dna_tokens, prot_tokens):
        """Return binding score + per-position importance + full attention matrices.

        Returns:
            score: (B,) binding probability [0,1]
            dna_importance: (B, L_dna) per-base importance
            prot_importance: (B, L_prot) per-residue importance
            dna2prot_attn: (B, L_dna, L_prot) full cross-attention matrix
            prot2dna_attn: (B, L_prot, L_dna) full cross-attention matrix
        """
        logits, all_weights = self.forward(dna_tokens, prot_tokens, return_weights=True)
        last_w = all_weights[-1]
        dna2prot = last_w["dna_to_prot"]    # (B, L_dna, L_prot) — DNA→Prot
        prot2dna = last_w["prot_to_dna"]   # (B, L_prot, L_dna) — Prot→DNA

        # DNA binding site: sum protein→DNA over protein residues
        #   = "how much does the entire protein attend to DNA base i?"
        dna_imp = prot2dna.sum(dim=1)       # (B, L_dna) — sum over protein dim

        # Protein DBD: sum DNA→protein over DNA bases
        #   = "how much does the entire DNA attend to protein residue j?"
        prot_imp = dna2prot.sum(dim=1)      # (B, L_prot) — sum over DNA dim

        return torch.sigmoid(logits), dna_imp, prot_imp, dna2prot, prot2dna
