# Shipped checkpoint

`nexusbind_at_esm2_t30/best_model.pt` — 10 MB, 90 state-dict entries.

## Training result

| | |
|---|---|
| Best epoch | 12 |
| Trainable params | 2,536,196 |
| Test (56 TFs, 230,782 samples, balanced) | **acc 0.9154 · AUC 0.9733 · AUPRC 0.9753 · MCC 0.8318** |

test row comes from the separate held-out evaluation.

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

## Loading

```python
import torch
ckpt = torch.load("checkpoints/nexusbind_at_esm2_t30/best_model.pt",
                  map_location="cpu", weights_only=False)
print(ckpt["hparams"], ckpt["val_metrics"])
```

The checkpoint stores the model weights but **not** the ESM2 backbone — the
protein branch is frozen and its embeddings are produced at inference time.
Only the trained projection layer (`prot_encoder.proj`) is stored, which is
how the required ESM2 dimension (640) is recovered.
