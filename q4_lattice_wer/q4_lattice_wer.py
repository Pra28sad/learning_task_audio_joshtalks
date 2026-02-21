#!/usr/bin/env python
"""
Q4: Lattice-based consensus pseudo-reference and fair WER computation.

Approach:
  1. Alignment unit: WORD-level (justified below).
  2. For each utterance, align all model outputs + reference using pairwise edit-distance
     alignment against a pivot (the reference), producing a word-level confusion network.
  3. At each position in the confusion network, apply majority voting:
     - If >= agreement_threshold models agree on a word AND it differs from the reference,
       trust the model consensus (the reference is likely wrong).
     - Otherwise, keep the reference word.
  4. Compute WER for each model against both the original reference and the
     lattice-based pseudo-reference.
  5. Models that were unfairly penalized (because the reference was wrong) should see
     reduced WER with the pseudo-reference. Others should stay the same or similar.

Why WORD-level alignment:
  - Hindi is a space-delimited language; word boundaries are clear.
  - Subword alignment introduces ambiguity and is harder to interpret.
  - Phrase-level alignment loses granularity for pinpointing specific errors.
  - WER itself is word-level, so alignment units match the evaluation metric.

Usage:
  python q4_lattice_wer.py --input_csv q4_data.csv --out_csv q4_results.csv

  Input CSV must have columns: reference, model_1, model_2, ..., model_N
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Edit-distance alignment (word-level)
# ---------------------------------------------------------------------------

EPSILON = "<eps>"  # Represents insertion/deletion in alignment


def align_sequences(ref: List[str], hyp: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """Align two word sequences using dynamic programming (Levenshtein).

    Returns a list of (ref_word, hyp_word) pairs where:
      - (word, word)  = match or substitution
      - (word, None)  = deletion (ref has it, hyp doesn't)
      - (None, word)  = insertion (hyp has it, ref doesn't)
    """
    n, m = len(ref), len(hyp)
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrace
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None))  # Deletion
            i -= 1
        else:
            alignment.append((None, hyp[j - 1]))  # Insertion
            j -= 1

    alignment.reverse()
    return alignment


# ---------------------------------------------------------------------------
# Confusion network construction (ROVER-style)
# ---------------------------------------------------------------------------

def build_confusion_network(
    reference: str,
    model_outputs: List[str],
) -> List[Dict[str, int]]:
    """Build a word-level confusion network by aligning each model against the reference.

    Returns a list of "slots" (one per alignment position). Each slot is a Counter
    mapping word -> vote_count. The reference word is included with a special key.
    Epsilon (insertion/deletion) is also tracked.
    """
    ref_tokens = norm_text(reference).split()
    model_token_lists = [norm_text(m).split() for m in model_outputs]

    # Align each model output against the reference
    all_alignments = []
    for model_tokens in model_token_lists:
        alignment = align_sequences(ref_tokens, model_tokens)
        all_alignments.append(alignment)

    # The reference alignment is trivially (ref[i], ref[i])
    # We need to merge all alignments into a single confusion network.
    # Use the reference as the backbone and accumulate votes per position.

    # Build the confusion network
    # Each position tracks votes for each word candidate
    cn: List[Dict[str, int]] = []

    # We align each model's alignment against the reference alignment.
    # Since all alignments are against the same reference, we can use
    # reference positions as anchors.

    # For each model alignment, extract the mapping from ref positions to hyp words
    for alignment in all_alignments:
        ref_pos = 0
        for ref_word, hyp_word in alignment:
            if ref_word is not None:
                # This corresponds to reference position ref_pos
                while len(cn) <= ref_pos:
                    cn.append(Counter())
                if hyp_word is not None:
                    cn[ref_pos][hyp_word] += 1
                else:
                    cn[ref_pos][EPSILON] += 1
                ref_pos += 1
            else:
                # Insertion: hyp has a word that ref doesn't
                # We attach it to the current ref position (before advancing)
                insert_pos = ref_pos  # or ref_pos - 1 if ref_pos > 0
                while len(cn) <= insert_pos:
                    cn.append(Counter())
                if hyp_word is not None:
                    cn[insert_pos][hyp_word] += 1

    # Ensure cn has at least len(ref_tokens) slots
    while len(cn) < len(ref_tokens):
        cn.append(Counter())

    return cn


# ---------------------------------------------------------------------------
# Consensus pseudo-reference
# ---------------------------------------------------------------------------

def consensus_pseudo_ref(
    reference: str,
    model_outputs: List[str],
    agreement_threshold: int = 3,
) -> str:
    """Build a consensus pseudo-reference using ROVER-style majority voting.

    At each position:
      - If >= agreement_threshold models agree on a word different from reference,
        use the model consensus (reference is likely wrong).
      - Otherwise, keep the reference word.

    This ensures:
      - Models unfairly penalized by a wrong reference get a corrected reference.
      - Models that genuinely made errors still get penalized correctly.
    """
    ref_tokens = norm_text(reference).split()

    if not ref_tokens:
        return reference

    # Build confusion network
    cn = build_confusion_network(reference, model_outputs)

    # Construct pseudo-reference using voting
    pseudo_tokens = []
    for i, ref_tok in enumerate(ref_tokens):
        if i < len(cn):
            votes = cn[i]
            if not votes:
                pseudo_tokens.append(ref_tok)
                continue

            # Find the most popular non-epsilon word
            word_votes = {k: v for k, v in votes.items() if k != EPSILON}
            if not word_votes:
                # All models deleted this position — still keep reference
                # unless overwhelming agreement
                if votes.get(EPSILON, 0) >= agreement_threshold:
                    continue  # Skip this token (models agree it shouldn't be there)
                pseudo_tokens.append(ref_tok)
                continue

            top_word, top_count = max(word_votes.items(), key=lambda x: x[1])

            if top_word != ref_tok and top_count >= agreement_threshold:
                # Models overwhelmingly agree on a different word
                pseudo_tokens.append(top_word)
            else:
                pseudo_tokens.append(ref_tok)
        else:
            pseudo_tokens.append(ref_tok)

    return " ".join(pseudo_tokens).strip()


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------

def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def wer(ref: str, hyp: str) -> float:
    r = norm_text(ref).split()
    h = norm_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_csv)

    # Auto-detect reference column: "Human" or "reference"
    if "Human" in df.columns:
        ref_col = "Human"
    elif "reference" in df.columns:
        ref_col = "reference"
    else:
        raise ValueError("Input CSV needs a 'Human' or 'reference' column")

    # Auto-detect model columns: "Model *" or "model_*"
    model_cols = [c for c in df.columns if c.startswith("Model ") or c.startswith("model_")]
    # Drop any empty/unnamed columns
    model_cols = [c for c in model_cols if c.strip()]
    if len(model_cols) < 2:
        raise ValueError(f"Need at least 2 model columns, found: {model_cols}")

    # Drop rows where reference or all models are NaN
    df = df.dropna(subset=[ref_col]).reset_index(drop=True)

    n_models = len(model_cols)
    print(f"Loaded {len(df)} utterances with {n_models} models: {model_cols}")
    print(f"Reference column: '{ref_col}'")
    print(f"Agreement threshold: {args.agreement_threshold} / {n_models}")

    base_sums: Dict[str, float] = {m: 0.0 for m in model_cols}
    lat_sums: Dict[str, float] = {m: 0.0 for m in model_cols}
    corrections = 0

    for _, row in df.iterrows():
        ref = str(row[ref_col])
        models = [str(row[m]) if pd.notna(row[m]) else "" for m in model_cols]
        pseudo_ref = consensus_pseudo_ref(ref, models, agreement_threshold=args.agreement_threshold)

        if norm_text(pseudo_ref) != norm_text(ref):
            corrections += 1

        for m in model_cols:
            hyp = str(row[m])
            base_sums[m] += wer(ref, hyp)
            lat_sums[m] += wer(pseudo_ref, hyp)

    n = len(df)
    out_rows = []
    for m in model_cols:
        base = 100.0 * base_sums[m] / n
        lat = 100.0 * lat_sums[m] / n
        delta = lat - base
        out_rows.append({
            "model": m,
            "wer_vs_reference_pct": round(base, 2),
            "wer_vs_lattice_pct": round(lat, 2),
            "delta_pct": round(delta, 2),
            "improved": "yes" if delta < -0.01 else "no",
        })

    out = pd.DataFrame(out_rows).sort_values("wer_vs_lattice_pct")
    out.to_csv(args.out_csv, index=False)

    # Print results
    print(f"\n{'='*75}")
    print(f"Lattice-Based WER Results")
    print(f"{'='*75}")
    print(f"Utterances: {n}")
    print(f"Reference corrections by lattice: {corrections}/{n} ({100*corrections/n:.1f}%)")
    print(f"\n{'Model':<15} {'WER(ref) %':<15} {'WER(lattice) %':<18} {'Delta %':<12} {'Improved'}")
    print(f"{'-'*15} {'-'*15} {'-'*18} {'-'*12} {'-'*8}")
    for _, r in out.iterrows():
        print(f"{r['model']:<15} {r['wer_vs_reference_pct']:<15} {r['wer_vs_lattice_pct']:<18} {r['delta_pct']:<12} {r['improved']}")
    print(f"{'='*75}")
    print(f"\nAlignment unit: WORD")
    print(f"Justification: Hindi is space-delimited; word boundaries are unambiguous.")
    print(f"  WER is defined at word level, so matching the alignment unit to the metric")
    print(f"  avoids granularity mismatch. Subword would add complexity without benefit")
    print(f"  for this task; phrase-level would lose ability to pinpoint specific errors.")
    print(f"\nSaved to: {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q4: Lattice-based WER with consensus pseudo-reference")
    ap.add_argument("--input_csv", required=True, help="CSV with reference + model_* columns")
    ap.add_argument("--out_csv", required=True, help="Output CSV with WER comparison")
    ap.add_argument("--agreement_threshold", type=int, default=3,
                     help="Min models that must agree to override reference (default: 3)")
    main(ap.parse_args())
