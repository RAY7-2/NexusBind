# Shipped checkpoint

`crossbind_at_esm2_t30/best_model.pt` — 10 MB, 90 state-dict entries.

## Provenance

This file was renamed for the release. It was copied verbatim from the
development repository at:

```
checkpoints/at_t30_protonly_cross2/best_model.pt
```

The run id encodes the experiment setup:

| token | meaning |
|---|---|
| `at` | trained on Arabidopsis DAP-seq data |
| `t30` | protein branch uses `facebook/esm2_t30_150M_UR50D` (640-dim) |
| `protonly` | DNA self-attention layers = 0, protein self-attention layers = 1 |
| `cross2` | 2 bidirectional cross-attention layers |

Note on the name: the evaluation run that produced the test numbers below
refers to a checkpoint at
`checkpoints/pretrained/at_t30_protonly_cross2_bs512_dp03_lr5/best_model.pt`
— i.e. the same run id with a `bs512_dp03_lr5` hyper-parameter suffix. The
shipped directory uses the short form. The two are treated as the same run
here because the hyper-parameters those suffixes encode (batch size 512,
dropout 0.3, lr 5e-4) match the shipped checkpoint's own `hparams` exactly.
The test metrics below are taken from that evaluation.

## Training result

| | |
|---|---|
| Best epoch | 12 |
| Trainable params | 2,536,196 |
| Validation (from the checkpoint itself) | acc 0.8842 · AUC 0.9676 · AUPRC 0.9685 |
| Test (56 TFs, 230,782 samples, balanced) | **acc 0.9154 · AUC 0.9733 · AUPRC 0.9753 · MCC 0.8318** |

The validation row is read off the shipped file (`ckpt["val_metrics"]`); the
test row comes from the separate held-out evaluation described above.

## Hyper-parameters (`ckpt["hparams"]`, verbatim)

```json
{
  "d_model": 128,
  "n_heads": 4,
  "d_ff": 256,
  "cross_attn_layers": 2,
  "dropout": 0.3,
  "lr": 0.0005,
  "batch_size": 512,
  "dna_encoder": "cnn",
  "protein_encoder": "esm2_emb"
}
```

`load_model()` in `scripts/inference.py` recovers the layer counts and
dimensions directly from the state dict rather than trusting this file, so
these values are documentation only.

## Loading

```python
import torch
ckpt = torch.load("checkpoints/crossbind_at_esm2_t30/best_model.pt",
                  map_location="cpu", weights_only=False)
print(ckpt["hparams"], ckpt["val_metrics"])
```

The checkpoint stores the model weights but **not** the ESM2 backbone — the
protein branch is frozen and its embeddings are produced at inference time.
Only the trained projection layer (`prot_encoder.proj`) is stored, which is
how the required ESM2 dimension (640) is recovered.
