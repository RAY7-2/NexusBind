"""
Config for NexusBind inference.

Holds the ESM2 model identity, the local ESM2 cache location, sequence limits
and the architectural defaults the shipped checkpoint was trained with.

The checkpoint is the source of truth for the architecture: ``load_model()`` in
inference.py reconstructs d_model / n_heads / d_ff / dropout / layer counts
from the checkpoint's own ``hparams`` dict and uses the values below only as
fallbacks.  See ``checkpoints/CHECKPOINT.md`` for the checkpoint this release
ships.
"""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# === Encoder selection ===
# DNA is always the one-hot CNN; protein is always frozen ESM2 embeddings.
DNA_ENCODER     = "cnn"
PROTEIN_ENCODER = "esm2_emb"

# === Pretrained model paths ===
ESM2_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
ESM2_CACHE_DIR  = PROJECT_ROOT / "pretrained_models" / "esm2"

# === Frozen encoders ===
FREEZE_DNA_ENCODER     = True
FREEZE_PROTEIN_ENCODER = True

# === DNA / Protein limits ===
DNA_MAX_LEN     = 200
DNA_VOCAB       = 4
DNA_NUC_TO_IDX  = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}

PROTEIN_MAX_LEN = 1024
PROTEIN_VOCAB   = 21
AA_TO_IDX       = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_TO_IDX["X"]  = 0

# === Model Architecture (fallbacks; the checkpoint overrides these) ===
D_MODEL              = 128
N_HEADS              = 4
D_FF                 = 256
DNA_SELF_ATTN_LAYERS = 1       # after DNA encoder, before cross-attn (0 = skip)
PROTEIN_SELF_ATTN_LAYERS = 1   # after protein encoder, before cross-attn
CROSS_ATTN_LAYERS    = 2
MLP_HIDDEN           = [256, 128]
DROPOUT              = 0.3
