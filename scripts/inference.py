#!/usr/bin/env python3
"""
CrossBind inference: binding score + binding-site prediction via cross-attention.

The protein is embedded on the fly with ESM2 (frozen), so the only external
asset needed is the ESM2 weights — see the README's ESM2 setup section.

Outputs:
  - Binding score
  - Per-position importance (summed cross-attention) for DNA and protein,
    each min-max scaled onto [0, 1], written to dna_importance.txt and
    prot_importance.txt (line N = position N, 1-based)
  - Full cross-attention weight matrices (raw, unnormalized)
  - result.json, including the top-1 position and the sequence window around it

All positions reported here are **1-based**.

Usage:
    python scripts/inference.py \
        --checkpoint checkpoints/crossbind_at_esm2_t30/best_model.pt \
        --protein-seq MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSL \
        --dna ATACGTAACGTACGTAGCTAGCTAGCTA

    # Offline, once the ESM2 weights are cached locally
    python scripts/inference.py ... --local-files-only \
        --esm2-cache-dir pretrained_models/esm2
"""

import sys, json, argparse, re
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from encoders import CNNEncoder, ESM2EmbEncoder, ESM2Encoder, DNA_CONV_BLOCKS
from model import TFBindingPretrainedModel


def select_device():
    """Prefer CUDA, then Apple Metal (MPS), and otherwise use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    """Load an esm2_emb checkpoint, inferring architecture from its state dict."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    hp = ckpt.get("hparams", {})
    state_dict = ckpt["model_state_dict"]

    d_model = hp.get("d_model", config.D_MODEL)
    n_heads = hp.get("n_heads", config.N_HEADS)
    d_ff = hp.get("d_ff", config.D_FF)
    dropout = hp.get("dropout", config.DROPOUT)

    def count_layers(prefix):
        indexes = set()
        pattern = re.compile(rf"^{re.escape(prefix)}\.encoder\.layers\.(\d+)\.")
        for key in state_dict:
            match = pattern.match(key)
            if match:
                indexes.add(int(match.group(1)))
        return max(indexes) + 1 if indexes else 0

    def count_cross_layers():
        indexes = set()
        pattern = re.compile(r"^cross_layers\.(\d+)\.")
        for key in state_dict:
            match = pattern.match(key)
            if match:
                indexes.add(int(match.group(1)))
        return max(indexes) + 1 if indexes else 0

    dna_self = count_layers("dna_self_attn")
    prot_self = count_layers("prot_self_attn")
    cross_layers = count_cross_layers()
    if cross_layers == 0:
        raise ValueError("Checkpoint contains no cross-attention layers and is not a CrossBind model.")

    proj_key = "prot_encoder.proj.weight"
    esm_dim = state_dict[proj_key].shape[1] if proj_key in state_dict else d_model

    print(f"  d={d_model}, heads={n_heads}, cross={cross_layers}, "
          f"dna_self={dna_self}, prot_self={prot_self}, esm_dim={esm_dim}")

    # DNA encoder (always CNN)
    dna_enc = CNNEncoder(config.DNA_VOCAB, DNA_CONV_BLOCKS, d_model, dropout)

    # Checkpoints store only the trained ESM2 projection, not the ESM2 backbone.
    prot_enc = ESM2EmbEncoder(d_model=d_model, esm_dim=esm_dim)

    model = TFBindingPretrainedModel(
        dna_encoder=dna_enc, protein_encoder=prot_enc,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dna_self_attn_layers=dna_self,
        protein_self_attn_layers=prot_self,
        cross_attn_layers=cross_layers, dropout=dropout,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, esm_dim


def extract_esm2_embedding_realtime(protein_seq, device, trained_proj, esm2_model,
                                    cache_dir, local_files_only=False):
    """Run real-time ESM2 + apply trained projection layer from model."""
    esm_enc = ESM2Encoder(model_name=esm2_model, d_model=trained_proj.out_features,
                          freeze=True, cache_dir=str(cache_dir),
                          local_files_only=local_files_only).to(device)
    if esm_enc.esm_dim != trained_proj.in_features:
        raise ValueError(
            f"ESM2 model '{esm2_model}' outputs {esm_enc.esm_dim} dimensions, but the "
            f"checkpoint expects {trained_proj.in_features}. Use the ESM2 model used in training."
        )

    # Tokenize.  Use AutoTokenizer directly instead of the project wrapper:
    # some transformers versions query the Hub for chat-template metadata even
    # when the tokenizer files are already cached.  A local-only retry avoids
    # that unnecessary network request after the model itself has loaded.
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            esm2_model, cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
    except OSError:
        if local_files_only:
            raise
        print("Tokenizer network lookup failed; retrying from the local ESM2 cache...")
        tok = AutoTokenizer.from_pretrained(
            esm2_model, cache_dir=str(cache_dir), local_files_only=True,
        )
    token_ids = np.asarray(
        tok(protein_seq, add_special_tokens=True, truncation=False)["input_ids"],
        dtype=np.int64,
    )
    L = min(len(token_ids), 1024)
    ids_t = torch.from_numpy(np.array(token_ids[:L])).unsqueeze(0).to(device)

    # ESM2 forward (use the underlying EsmModel for raw output)
    with torch.no_grad():
        out = esm_enc.esm(input_ids=ids_t, attention_mask=(ids_t != 0))
        # out is a HuggingFace BaseModelOutput, not a tuple
        raw_emb = out.last_hidden_state.float()
        proj_emb = trained_proj(raw_emb)

    del esm_enc
    return proj_emb, L


def tokenize_dna(seq: str, max_len=None):
    """Tokenize DNA → one-hot + mask."""
    max_len = max_len or config.DNA_MAX_LEN
    mapping = config.DNA_NUC_TO_IDX
    seq = re.sub(r"\s+", "", seq.upper())
    invalid = sorted(set(seq) - set(mapping))
    if not seq:
        raise ValueError("DNA sequence is empty.")
    if invalid:
        raise ValueError(f"DNA sequence contains unsupported character(s): {', '.join(invalid)}. "
                         "Allowed characters: A, C, G, T, N.")
    if len(seq) > max_len:
        print(f"WARNING: DNA is {len(seq)} bp; truncating to the first {max_len} bp.")
    tokens = [mapping[c] for c in seq]
    L = min(len(tokens), max_len)
    # Match training collation: cap the sequence length, but do not add
    # inference-only trailing padding for a single DNA sequence.
    arr = np.zeros((1, config.DNA_VOCAB, L), dtype=np.float32)
    mask = np.ones((1, L), dtype=bool)
    for i in range(L):
        arr[0, tokens[i], i] = 1.0
    mask[0, :L] = True
    return torch.from_numpy(arr), torch.from_numpy(mask), L


def validate_protein_sequence(seq: str) -> str:
    """Normalize a raw amino-acid sequence and reject unsupported residue codes."""
    seq = re.sub(r"\s+", "", seq.upper())
    allowed = set("ACDEFGHIKLMNPQRSTVWYX")
    invalid = sorted(set(seq) - allowed)
    if not seq:
        raise ValueError("Protein sequence is empty.")
    if invalid:
        raise ValueError(f"Protein sequence contains unsupported character(s): {', '.join(invalid)}. "
                         "Allowed amino acids are the 20 standard residues plus X.")
    return seq


def normalize_importance(values):
    """Min-max scale an importance vector onto [0, 1].

    A constant vector has no meaningful peak, so it maps to all zeros rather
    than dividing by zero.
    """
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(np.asarray(values, dtype=np.float64))
    return (np.asarray(values, dtype=np.float64) - lo) / (hi - lo)


def sequence_window(seq, pos0, flank=10):
    """Return seq[pos0] plus `flank` characters either side, clamped to the ends.

    `pos0` is a 0-based index into `seq`; the returned string is what the
    caller reports to the user, so it may be shorter than 2*flank+1 near the
    sequence boundaries.
    """
    start = max(0, pos0 - flank)
    end = min(len(seq), pos0 + flank + 1)
    return seq[start:end]


def run_inference(model, dna_tokens, dna_mask, prot_tokens, device):
    """
    Run full inference, returning score + all attention matrices.

    Returns:
        score:        float, binding probability [0,1]
        dna_imp:      (L_dna,) per-base importance (raw; caller normalizes)
        prot_imp:     (L_prot,) per-residue importance (raw; caller normalizes)
        dna2prot_attn: (L_dna, L_prot) full cross-attention matrix (raw weights)
        prot2dna_attn: (L_prot, L_dna) full cross-attention matrix (raw weights)
    """
    with torch.no_grad():
        logits, all_weights = model(dna_tokens, prot_tokens, return_weights=True)

    score = torch.sigmoid(logits).item()
    w = all_weights[-1]  # last cross-attention layer

    dna2prot = w["dna_to_prot"][0].cpu().numpy()     # (L_dna, L_prot)
    prot2dna = w["prot_to_dna"][0].cpu().numpy()     # (L_prot, L_dna)

    # DNA binding site: how much does protein attend to each DNA base?
    dna_imp = prot2dna.sum(axis=0)                   # (L_dna,) sum over protein residues
    # Protein DBD: how much does DNA attend to each protein residue?
    prot_imp = dna2prot.sum(axis=0)                  # (L_prot,) sum over DNA bases

    return score, dna_imp, prot_imp, dna2prot, prot2dna


def _plot_importance(values, xlabel, ylabel, title, out_path, color):
    """Plot per-position importance as a curve with a diamond at every position.

    Positions on the x axis are 1-based, matching every other reported
    coordinate (the .txt files, result.json and scan_promoter's TSV).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    L = len(values)
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(1, L + 1)                       # 1-based positions

    ax.plot(x, values, color=color, linewidth=0.8,
            marker='D', markersize=4, markeredgewidth=0)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(1, max(L, 2))
    ax.grid(axis='y', alpha=0.3)

    if L:
        # A little headroom so the topmost diamond isn't clipped
        ax.set_ylim(0, max(float(np.max(values)), 1e-6) * 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    # Protein input
    parser.add_argument("--protein-seq", type=str, required=True,
                        help="Raw protein sequence (embedded with ESM2)")
    parser.add_argument("--esm2-model", type=str, default=config.ESM2_MODEL_NAME,
                        help="HuggingFace ESM2 model (must match the checkpoint's embedding dimension)")
    parser.add_argument("--esm2-cache-dir", type=Path, default=config.ESM2_CACHE_DIR,
                        help="Directory used to cache the ESM2 model and tokenizer")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load ESM2 only from --esm2-cache-dir; never access the network")
    # DNA input
    parser.add_argument("--dna", type=str, required=True,
                        help="DNA sequence (the model was trained on 200 bp windows)")
    parser.add_argument("--dna-max-len", type=int, default=config.DNA_MAX_LEN,
                        help=f"Maximum DNA length (default: training length {config.DNA_MAX_LEN})")
    # Output
    parser.add_argument("--output-dir", type=str, default="results/inference")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Task folder name (default: custom_protein_manual)")
    args = parser.parse_args()

    if args.dna_max_len < 1:
        parser.error("--dna-max-len must be at least 1")

    device = select_device()
    print(f"Device: {device}")

    dna_seq = args.dna
    dna_source = "manual"

    # ── Load model (always uses ESM2EmbEncoder, matching checkpoint) ──
    model, _ = load_model(args.checkpoint, device)

    # ── Prepare protein input ──
    # Use the trained proj from the model, then set the model's
    # prot_encoder.proj to Identity to avoid double-projection.
    prot_seq = validate_protein_sequence(args.protein_seq)
    print(f"\nExtracting ESM2 embedding for protein sequence...")
    print(f"  {prot_seq[:60]}...")

    trained_proj = model.prot_encoder.proj  # trained Linear(esm_dim → d_model)
    model.prot_encoder.proj = torch.nn.Identity()  # bypass for forward

    prot_tokens, prot_len = extract_esm2_embedding_realtime(
        prot_seq, device, trained_proj, args.esm2_model,
        args.esm2_cache_dir, args.local_files_only)
    prot_tokens = prot_tokens.to(device)
    prot_source = "realtime_esm2"
    protein_label = "custom_protein"
    print(f"  Extracted: {prot_len - 2} amino acids (+ <cls>/<eos>)")

    # ── Tokenize DNA ──
    dna_tokens, dna_mask, dna_len = tokenize_dna(dna_seq, args.dna_max_len)
    dna_tokens, dna_mask = dna_tokens.to(device), dna_mask.to(device)

    print(f"DNA:     {dna_len} bp (source: {dna_source})")
    print(f"         5' {dna_seq[:60]}... 3'")

    # ── Run inference ──
    print("\nRunning inference...")
    score, dna_imp, prot_imp, dna2prot, prot2dna = run_inference(
        model, dna_tokens, dna_mask, prot_tokens, device)

    # Raw ESM2 inputs contain <cls> and <eos>. Remove them before reporting
    # protein importance so positions map directly to amino-acid indexes.
    prot_imp = prot_imp[1:-1]
    dna2prot = dna2prot[:, 1:-1]
    prot2dna = prot2dna[1:-1, :]
    prot_len = len(prot_seq)

    # Min-max scale each profile onto [0, 1] *after* trimming the special
    # tokens, so the reported peak is exactly 1.0 over the positions we show.
    valid_dna = normalize_importance(dna_imp[:dna_len])
    valid_prot = normalize_importance(prot_imp[:prot_len])

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"Binding score: {score:.4f}  →  {'BIND' if score > 0.5 else 'NOT BIND'}")
    print(f"DNA importance:     min-max scaled to [0,1] over {dna_len} bp")
    print(f"Protein importance: min-max scaled to [0,1] over {prot_len} aa")
    print(f"Attention matrices: dna2prot {dna2prot.shape}, prot2dna {prot2dna.shape}")

    # Top DNA positions (reported 1-based)
    top_dna = np.argsort(valid_dna)[-10:][::-1]
    print(f"\nTop 10 DNA positions (potential binding site):")
    for i, pos in enumerate(top_dna):
        print(f"  #{i+1}: pos={pos+1:4d}  score={valid_dna[pos]:.4f}  "
              f"'{sequence_window(dna_seq, pos)}'")

    # Top Protein positions (reported 1-based)
    top_prot = np.argsort(valid_prot)[-10:][::-1]
    print(f"\nTop 10 Protein positions (potential DBD):")
    for i, pos in enumerate(top_prot):
        aa = prot_seq[pos] if pos < len(prot_seq) else '?'
        print(f"  #{i+1}: pos={pos+1:4d}  score={valid_prot[pos]:.4f}  AA='{aa}'")

    # ── Save output ──
    out_dir = Path(args.output_dir)
    run_name = args.run_name or f"{protein_label}_{dna_source}"
    task_dir = out_dir / run_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Per-position importance. Written as plain text only — line N holds the
    # value for position N (1-based), so the file is directly readable.
    np.savetxt(task_dir / "dna_importance.txt", valid_dna, fmt='%.6f')
    np.savetxt(task_dir / "prot_importance.txt", valid_prot, fmt='%.6f')

    # Full attention matrices (raw attention weights, not normalized)
    np.save(task_dir / "dna2prot_attn.npy", dna2prot[:dna_len, :prot_len].astype(np.float32))
    np.save(task_dir / "prot2dna_attn.npy", prot2dna[:prot_len, :dna_len].astype(np.float32))

    # ── Plots ──
    _plot_importance(valid_dna, "DNA Position (bp)", "Binding Importance",
                     f"DNA Binding Site Prediction\n{run_name} | score={score:.3f}",
                     task_dir / "dna_importance.png", color='#2196F3')
    _plot_importance(valid_prot, "Protein Position (aa)", "Binding Importance",
                     f"Protein DBD Prediction\n{run_name} | score={score:.3f}",
                     task_dir / "prot_importance.png", color='#2E7D32')

    # Summary JSON
    top_dna_pos = int(top_dna[0])
    top_prot_pos = int(top_prot[0])
    summary = {
        "run_name": run_name,
        "coordinate_base": 1,
        "protein_label": protein_label,
        "dna_source": dna_source,
        "protein_source": prot_source,
        "dna_length": dna_len,
        "protein_length": prot_len,
        "binding_score": float(score),
        "prediction": "BIND" if score > 0.5 else "NOT BIND",
        "dna_top1": {
            "position": top_dna_pos + 1,
            "sequence": sequence_window(dna_seq, top_dna_pos),
        },
        "prot_top1": {
            "position": top_prot_pos + 1,
            "sequence": sequence_window(prot_seq, top_prot_pos),
        },
    }
    with open(task_dir / "result.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved: {task_dir}/")
    for fname in ["dna_importance.txt", "prot_importance.txt",
                  "dna2prot_attn.npy", "prot2dna_attn.npy",
                  "dna_importance.png", "prot_importance.png", "result.json"]:
        print(f"  {fname}")


if __name__ == "__main__":
    main()
