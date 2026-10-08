# Promoter example

`dxs2-promoter.fa` — the *DXS2* promoter, 556 bp.

```
>dxs2_promoter
TTATTAATTAACCTATGAACATATTTTATATGTATGTGTTATATGTATATCTATGTTT...
```

## Run it

```bash
python scripts/scan_promoter.py \
    --checkpoint checkpoints/nexusbind_at_esm2_t30/best_model.pt \
    --protein-seq "$(grep -v '>' examples/protein/mnb1a-protein-seq.faa | tr -d '\n')" \
    --promoter-file examples/promoter/dxs2-promoter.fa \
    --output-dir results/scan_promoter --run-name dxs2_mnb1a
```

556 bp at the default `stride 1` gives **357 windows**. The highest-scoring
window is the predicted binding site; `window_scores.csv` lists every window
with its score and attention peak.

**The output of this exact command is committed at `scan_dxs2_mnb1a/`**, so you
can inspect it before running anything:

| | |
|---|---|
| windows | 357 |
| top window | index **12** (bases **12–211**), score **0.993** |
| attention peak | base **109** of the promoter (local 98) |
| score range across all windows | 0.000 – 0.993, mean 0.014 |

The sharp contrast between peak and mean is the point: nearly all 357 windows
score near zero and one stands out. Note that scoring only the *first* 200 bp
gives 0.03 — the binding site is not at the start, which is exactly why a
whole-promoter scan is the right tool here.

### What's in `scan_dxs2_mnb1a/`

| file | contents |
|---|---|
| `window_scores.csv` | one row per window — the human-readable result |
| `result.json` | summary: window count, max/mean score, top window index |
| `window_attention_importance.npy` | per-base importance profile for every window |
| `scan_config.json` | the exact settings used for this run |

At 372 KB total this is small enough to commit; a real genome-wide scan is not,
which is why nothing under `results/` is tracked.

### `window_scores.csv`

| column | meaning |
|---|---|
| `window_index` | 1-based window number |
| `start` / `end` | 1-based, inclusive bounds of the window in the input |
| `sequence` | the 200 bp window |
| `binding_score` | sigmoid output, `0.0`–`1.0` |
| `attention_peak_local` | 1-based position of the attention peak **within** the window |
| `attention_peak_input` | 1-based position **in the input**; equals `start + attention_peak_local - 1` |

### `window_attention_importance.npy`

`(357, 200)` float32 — the per-base DNA importance profile *inside* each
window. The CSV answers "which window?"; this array answers "what did the
profile look like in it?".

- Row *i* is window `window_index` *i+1*
- Entry *j* of row *i* is **1-based base `start + j`** in the input, where
  `start` is that window's `start` column (so `j=0` is the first base of the
  window). Equivalently it is `sequence[j]` of that window's CSV row
- Values come from the last cross-attention layer's Protein→DNA attention,
  summed over protein residues
- **Each row is min-max scaled onto [0, 1] independently**, matching how
  `inference.py` reports its DNA profile. Every row therefore has max 1.0 at
  its peak and min 0.0 — the argmax of row *i* is exactly the CSV's
  `attention_peak_local` for that window

Use it to tell a single sharp peak from several broad ones:

```python
import numpy as np
imp = np.load("window_attention_importance.npy")     # (357, 200)

row = imp[11]                                        # window_index 12
peak = row.argmax()                                  # 97 -> CSV says 98 (1-based)
print(row[peak - 10:peak + 11])                      # profile around the peak
print((row > 0.5).sum())                             # 0.5 = half the peak height
```

Attention is meant to be read **within** a window — it says where along that
200 bp the protein is looking, not how strongly a window binds. To rank
windows, use `binding_score` in the CSV.

### Coordinates are 1-based

Every position and index this tool reports is **1-based**: the first base is
position 1, the first window is `window_index` 1, and `end` is *inclusive*
(so a 200 bp window starting at 12 spans 12–211). `result.json` records
`"coordinate_base": 1` so a downstream reader never has to guess.

## Following up on the top window

The scan tells you *which* window binds; `inference.py` gives the full
position-level breakdown for that one window — including the protein side,
which the scan does not report.

**The output is committed at `inference_dxs2_mnb1a/`.** It scores the top
window from the scan (bases 12–211, i.e. `promoter[11:211]`):

```bash
python scripts/inference.py \
    --checkpoint checkpoints/nexusbind_at_esm2_t30/best_model.pt \
    --protein-seq "$(grep -v '>' examples/protein/mnb1a-protein-seq.faa | tr -d '\n')" \
    --dna "$(python3 -c "
p=''.join(l.strip() for l in open('examples/promoter/dxs2-promoter.fa').read().splitlines()[1:])
print(p[11:211])")" \
    --run-name inference_dxs2_mnb1a --output-dir examples/promoter
```

| | scan (`scan_dxs2_mnb1a/`) | inference (`inference_dxs2_mnb1a/`) |
|---|---|---|
| binding score | 0.9932180 | 0.9932180 |
| DNA peak | local 98 → promoter 109 | local 98 |
| protein DBD | *not reported* | peak at residue **139** |

The two agree exactly, which is a useful sanity check that both entry points
run the same model the same way. The protein side is where inference adds
value: the DNA→Protein attention localizes the DNA-binding domain to a handful
of residues (139, 79, 78, 140, 71, 51 — mostly arginines), and
`prot_importance.png` plots the profile across the full 238-residue protein.

Note the coordinate shift: `attention_peak_local` is relative to the *window*,
so promoter position = `start + attention_peak_local - 1` (12 + 98 − 1 = 109),
which is what the scan's `attention_peak_input` column already gives you.

### What's in `inference_dxs2_mnb1a/`

| file | contents |
|---|---|
| `result.json` | score, prediction, lengths, and the top-1 DNA and protein hits |
| `dna_importance.txt` | 200 values, one per line — line N is position N (1-based) |
| `prot_importance.txt` | 238 values, one per line — line N is residue N (1-based) |
| `dna_importance.png` | blue curve, diamond marker at every base |
| `prot_importance.png` | green curve, diamond marker at every residue |
| `dna2prot_attn.npy` | `(200, 238)` full attention matrix, raw weights |
| `prot2dna_attn.npy` | `(238, 200)` full attention matrix, raw weights |

Both importance profiles are **min-max scaled onto [0, 1]** after the ESM2
`<cls>`/`<eos>` tokens are removed, so the reported peak is exactly 1.0 and
the value at each position is readable straight off the txt file. The two
`*_attn.npy` matrices hold the **raw** attention weights, not the scaled
profile. Top-1 hits:

```json
"dna_top1":   {"position": 98,  "sequence": "ACAGAAAAAAAAGTAGTTTAG"},
"prot_top1":  {"position": 139, "sequence": "RVVAPAPELATAADPGKTATT"}
```

Each `sequence` is the peak position plus 10 characters either side (21 total),
clamped at the sequence ends.

## Your own promoter

Replace the file, or pass a raw sequence inline with `--promoter`:

- Single-line raw sequence of `A`/`C`/`G`/`T`/`N` (case-insensitive)
- Or a FASTA file with **one** record — multi-record FASTA is not supported

Whitespace and newlines are stripped. The sequence must be **at least 200 bp**
(the window size the model was trained on).

## Notes

- `--window-size` must stay at 200 — the trained model has no other input size.
- Use `--stride N` to step more coarsely on very long inputs.
- Attention is reported **within** each window. Its job is to say *where* along
  those 200 bp the protein is looking — it is not a measure of how strongly a
  window binds. Rank windows by `binding_score`.
- See the repository README for how to point `--esm2-cache-dir` at a local
  ESM2 download and add `--local-files-only` to run fully offline.
