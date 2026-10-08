# Protein example

`mnb1a-protein-seq.faa` — the *mnb1a* protein, 238 aa.

```
>mnb1a
MQEASSAAAAGAEPGRRAAQHQFAGVDLRRPKGYAAPAPAPAVGEGDPCPRCASRDT...
```

## Run it

```bash
python scripts/inference.py \
    --checkpoint checkpoints/nexusbind_at_esm2_t30/best_model.pt \
    --protein-seq "$(grep -v '>' examples/protein/mnb1a-protein-seq.faa | tr -d '\n')" \
    --dna "$(grep -v '>' examples/promoter/dxs2-promoter.fa | tr -d '\n')" \
    --run-name mnb1a_example
```

> The DXS2 promoter is 556 bp, so it is truncated to the model's 200 bp training
> window (a warning is printed) and the score comes back **0.03 — NOT BIND**.
> That is expected: the binding site is not in the first 200 bp. Use
> `scripts/scan_promoter.py`, which slides a 200 bp window across the whole
> promoter and finds the 0.993 peak at bases 12–211. To score a specific
> region, slice it yourself — see the committed example at
> [`../promoter/inference_dxs2_mnb1a/`](../promoter/inference_dxs2_mnb1a/),
> which does exactly that on the top-scoring window.

## Your own protein

Replace the file, or pass a raw sequence inline with `--protein-seq`:

- Single-line raw sequence of uppercase amino acids
- Or a FASTA file with **one** record — multi-record FASTA is not supported

Allowed residues are the 20 standard amino acids plus `X` for unknown. The
residue-importance output is aligned position-by-position with your input, so
one sequence per run.

## Notes

- `--dna` must be supplied here as well. The model was trained on **200 bp**
  windows; longer input is truncated (with a warning) and shorter input is
  accepted but is out of the training distribution.
- See the repository README for how to point `--esm2-cache-dir` at a local
  ESM2 download and add `--local-files-only` to run fully offline.
