#!/usr/bin/env python3
"""Scan a long promoter with the current CrossBind inference interface.

Every complete 200-bp window is scored with stride 1 by default.  Outputs
include one TSV row per window, a score vector, and a DNA-attention matrix.
Attention is calculated from the last cross-attention layer as the sum of
protein-to-DNA attention over protein positions.

All positions and window indexes in the output are **1-based**.

Examples:
  python scripts/scan_promoter.py \
      --checkpoint checkpoints/crossbind_at_esm2_t30/best_model.pt \
      --protein-seq 'M...' --promoter-file promoter.fa \
      --output-dir results/promoter_scan --run-name my_promoter

  # Offline, once the ESM2 weights are cached locally
  python scripts/scan_promoter.py ... --local-files-only
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config
from inference import (
    extract_esm2_embedding_realtime,
    load_model,
    normalize_importance,
    select_device,
    validate_protein_sequence,
)


def read_sequence(path):
    chunks = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(">"):
                chunks.append(line)
    sequence = "".join(chunks).upper()
    invalid = sorted(set(sequence) - set(config.DNA_NUC_TO_IDX))
    if not sequence:
        raise ValueError(f"No DNA sequence found in {path}.")
    if invalid:
        raise ValueError(f"Invalid DNA base(s) in {path}: {', '.join(invalid)}")
    return sequence


def normalize_sequence(sequence):
    sequence = "".join(sequence.upper().split())
    invalid = sorted(set(sequence) - set(config.DNA_NUC_TO_IDX))
    if not sequence or invalid:
        raise ValueError(f"Invalid promoter sequence; invalid bases: {invalid}")
    return sequence


def encode_dna_batch(sequences):
    length = len(sequences[0])
    if any(len(seq) != length for seq in sequences):
        raise ValueError("All DNA windows must have the same length.")
    indices = np.asarray(
        [[config.DNA_NUC_TO_IDX[base] for base in seq] for seq in sequences],
        dtype=np.int64,
    )
    onehot = np.eye(config.DNA_VOCAB, dtype=np.float32)[indices]
    return torch.from_numpy(onehot.transpose(0, 2, 1))


def load_protein(args, model, device):
    """Embed the protein with ESM2, using the checkpoint's trained projection.

    Sets the model's own ``prot_encoder.proj`` to Identity so the embedding is
    not projected twice.
    """
    sequence = validate_protein_sequence(args.protein_seq)
    projection = model.prot_encoder.proj
    model.prot_encoder.proj = torch.nn.Identity()
    tokens, length = extract_esm2_embedding_realtime(
        sequence, device, projection, args.esm2_model,
        args.esm2_cache_dir, args.local_files_only,
    )
    return tokens.to(device), length, "realtime_esm2", "custom_protein"


def write_results(output_dir, run_name, promoter, rows, scores, attention, peaks):
    """Write per-window results.

    All positions and indexes are **1-based**, matching inference.py: the first
    window is `window_index` 1, the first base is `start` 1, and `end` is
    inclusive.  Internally `rows` still carries 0-based offsets, so each value
    written here is offset by one.

    Everything tabular goes into a single `window_scores.csv`; the remaining
    arrays that do not fit a row-per-window table are saved as `.npy`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "window_scores.csv", "w", newline="") as handle:
        writer = csv.writer(handle)          # comma-delimited
        writer.writerow([
            "window_index", "start", "end",
            "sequence", "binding_score", "attention_peak_local",
            "attention_peak_input",
        ])
        for row, score, importance, peak in zip(rows, scores, attention, peaks):
            start_0 = row[1]
            writer.writerow([
                row[0] + 1,                 # 1-based window index
                start_0 + 1,                # 1-based first base of the window
                start_0 + row[2] - row[1],  # 1-based inclusive last base
                row[3], f"{score:.7f}",
                peak[0] + 1,                # 1-based position within the window
                start_0 + peak[0] + 1,      # 1-based position in the input
            ])

    # Per-base DNA importance for every window: (n_windows, window_size).
    # Each row is min-max scaled onto [0, 1] independently, matching how
    # inference.py reports its DNA profile.  Scaling is monotonic, so the
    # argmax of each row is unchanged and still agrees with the CSV.
    np.save(output_dir / "window_attention_importance.npy",
            np.asarray([normalize_importance(row) for row in attention],
                       dtype=np.float32))

    summary = {
        "run_name": run_name,
        "coordinate_base": 1,
        "promoter_length": len(promoter),
        "window_size": 200,
        "stride": 1,
        "n_windows": len(rows),
        "max_score": float(np.max(scores)),
        "max_score_window_index": int(np.argmax(scores)) + 1,
        "mean_score": float(np.mean(scores)),
        "score_matrix_shape": [len(scores)],
        "attention_matrix_shape": list(np.asarray(attention).shape),
        "files": {
            "windows": "window_scores.csv",
            "attention": "window_attention_importance.npy",
        },
    }
    with open(output_dir / "result.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Scan promoter windows with CrossBind.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protein-seq", required=True,
                        help="Raw protein sequence (embedded with ESM2)")
    dna = parser.add_mutually_exclusive_group(required=True)
    dna.add_argument("--promoter")
    dna.add_argument("--promoter-file", type=Path)
    parser.add_argument("--esm2-model", default=config.ESM2_MODEL_NAME)
    parser.add_argument("--esm2-cache-dir", type=Path, default=config.ESM2_CACHE_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scan_promoter"))
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.window_size != config.DNA_MAX_LEN:
        parser.error(f"--window-size must be {config.DNA_MAX_LEN} for the trained model.")
    if args.stride < 1 or args.batch_size < 1:
        parser.error("--stride and --batch-size must be at least 1.")
    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    promoter = normalize_sequence(args.promoter if args.promoter else read_sequence(args.promoter_file))
    if len(promoter) < args.window_size:
        parser.error("Promoter must be at least 200 bp.")

    device = select_device()
    model, _ = load_model(args.checkpoint, device)
    prot_tokens, prot_len, protein_source, protein_label = load_protein(
        args, model, device
    )
    n_windows = (len(promoter) - args.window_size) // args.stride + 1
    rows = [
        (index, start, start + args.window_size, promoter[start:start + args.window_size])
        for index, start in enumerate(range(0, len(promoter) - args.window_size + 1, args.stride))
    ]
    scores, importances, peaks = [], [], []
    with torch.no_grad():
        for first in range(0, len(rows), args.batch_size):
            block = rows[first:first + args.batch_size]
            dna_tokens = encode_dna_batch([row[3] for row in block]).to(device)
            protein_tokens = prot_tokens.expand(len(block), -1, -1)
            logits, weights = model(dna_tokens, protein_tokens, return_weights=True)
            block_scores = torch.sigmoid(logits).cpu().numpy()
            # Last cross-attention layer only: Protein->DNA summed over residues
            # gives the per-base DNA importance profile for this window.
            block_importance = weights[-1]["prot_to_dna"].sum(axis=1).cpu().numpy()
            for score, importance in zip(block_scores, block_importance):
                position = int(np.argmax(importance))
                peaks.append((position, float(importance[position])))
                scores.append(float(score))
            importances.extend(block_importance)
            if first == 0 or first + len(block) == len(rows):
                print(f"Processed {first + len(block)}/{len(rows)} windows")

    out = args.output_dir / args.run_name
    write_results(out, args.run_name, promoter, rows, scores, importances, peaks)
    with open(out / "scan_config.json", "w") as handle:
        json.dump({"checkpoint": str(args.checkpoint), "protein": protein_label,
                   "protein_source": protein_source, "protein_tokens": prot_len,
                   "device": str(device), "window_size": args.window_size,
                   "stride": args.stride, "batch_size": args.batch_size}, handle, indent=2)


if __name__ == "__main__":
    main()
