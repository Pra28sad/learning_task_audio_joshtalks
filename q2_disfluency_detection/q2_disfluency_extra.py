#!/usr/bin/env python
"""
Q2: Detect disfluencies at segment level and export clips + structured sheet.

Usage:
  # Requires: manifests from Q1 preprocess + downloaded audio
  python q2_disfluency_extra.py \
      --manifest data_q1/manifest_all.csv \
      --audio_dir data_q1/audio \
      --out_dir data_q2_disfluency

Deliverables produced:
  - data_q2_disfluency/disfluency_sheet.csv   (structured sheet)
  - data_q2_disfluency/clips/*.wav             (segmented audio clips)
  - data_q2_disfluency/methodology.txt         (methodology summary)
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Disfluency detection patterns
# ---------------------------------------------------------------------------

# Hindi fillers: common filler words in Hindi conversational speech
FILLERS_EXACT = {
    "अ", "आ", "अं", "अम्म", "उह", "हम्म", "हम्मम", "हम्म्म", "हूं", "हूँ",
    "हुन", "ह्यू", "ह्म्म", "ह्म्म्म", "हं", "हां", "हाँ",
}

# Fillers that are context-dependent (only fillers when standalone / sentence-initial)
FILLERS_CONTEXTUAL = {
    "मतलब", "तो", "बस", "वो", "ये", "अच्छा", "ओके", "हा",
}

# Pattern for prolongation in text (repeated characters like "सोऊऊऊ")
PROLONGATION_RE = re.compile(r"(.)\1{2,}")

# Pattern for false starts: words ending with hyphen, or ellipsis
FALSE_START_RE = re.compile(r"\b\w+\s*[-–—]\s|\.\.\.|…")

# Repetition: consecutive identical words
def _find_repetitions(tokens: List[str]) -> List[Tuple[int, int, str]]:
    """Find runs of repeated consecutive words. Returns (start_idx, end_idx, word)."""
    reps = []
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == tokens[i + 1]:
            j = i + 1
            while j < len(tokens) and tokens[j] == tokens[i]:
                j += 1
            reps.append((i, j - 1, tokens[i]))
            i = j
        else:
            i += 1
    return reps


def classify_disfluencies(text: str) -> List[Dict[str, str]]:
    """Detect disfluencies in a text segment.

    Returns a list of dicts with keys: type, evidence.
    Each entry represents one detected disfluency occurrence.
    """
    t = norm(text)
    if not t:
        return []

    findings = []
    tokens = t.split()

    # 1. Filler detection
    for tok in tokens:
        clean = re.sub(r"[।,.\s]", "", tok)
        if clean in FILLERS_EXACT:
            findings.append({"type": "filler", "evidence": clean})

    # Contextual fillers: only when they appear at sentence boundaries or standalone
    for tok in tokens:
        clean = re.sub(r"[।,.\s]", "", tok)
        if clean in FILLERS_CONTEXTUAL:
            idx = tokens.index(tok)
            # Filler if at start, end, or surrounded by punctuation-like tokens
            if idx == 0 or idx == len(tokens) - 1:
                findings.append({"type": "filler", "evidence": clean})

    # 2. Repetition detection
    reps = _find_repetitions(tokens)
    for start_idx, end_idx, word in reps:
        count = end_idx - start_idx + 1
        findings.append({
            "type": "repetition",
            "evidence": f"{word} (x{count})",
        })

    # 3. False starts (interrupted words, ellipsis)
    for m in FALSE_START_RE.finditer(t):
        findings.append({"type": "false_start", "evidence": m.group().strip()})

    # 4. Prolongation (repeated characters in text)
    for m in PROLONGATION_RE.finditer(t):
        findings.append({"type": "prolongation", "evidence": m.group()})

    # 5. Hesitation: very short segments with backchannel words
    hesitation_markers = {"हां", "हाँ", "हा", "जी", "हूं", "हूँ", "हुं", "हुन", "ह्यू"}
    if len(tokens) <= 3 and all(re.sub(r"[।,.\s]", "", tok) in hesitation_markers for tok in tokens):
        findings.append({"type": "hesitation", "evidence": t})

    return findings


# ---------------------------------------------------------------------------
# Audio clipping
# ---------------------------------------------------------------------------

def clip_with_ffmpeg(audio_path: Path, out_path: Path, start: float, end: float) -> bool:
    """Clip a segment from audio file using ffmpeg. Returns True on success."""
    import subprocess

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", "16000",
        "-loglevel", "error",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.manifest)

    # Ensure required columns
    for col, default in [("segment_id", 0), ("start", None), ("end", None)]:
        if col not in df.columns:
            df[col] = default

    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    rows: List[Dict[str, object]] = []
    clips_created = 0

    print(f"Scanning {len(df)} segments for disfluencies ...")

    for _, r in df.iterrows():
        text = norm(str(r.get("text", "")))
        if not text:
            continue

        findings = classify_disfluencies(text)
        if not findings:
            continue

        rec_id = int(r["recording_id"])
        seg_id = int(r["segment_id"])
        start = r.get("start")
        end = r.get("end")

        # Clip audio if timestamps available
        clip_path = ""
        if pd.notna(start) and pd.notna(end):
            src = Path(args.audio_dir) / f"{rec_id}.wav"
            dst = clips_dir / f"{rec_id}_seg{seg_id}.wav"
            if src.exists():
                if clip_with_ffmpeg(src, dst, float(start), float(end)):
                    clip_path = str(dst)
                    clips_created += 1

        # One row per disfluency occurrence (as required by deliverables)
        for f in findings:
            rows.append({
                "recording_id": rec_id,
                "segment_id": seg_id,
                "start_sec": start,
                "end_sec": end,
                "disfluency_type": f["type"],
                "evidence": f["evidence"],
                "segment_text": text,
                "clip_path": clip_path,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "disfluency_sheet.csv"
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv, index=False)

    # Print summary
    print(f"\nDisfluency Detection Summary:")
    print(f"  Total segments scanned: {len(df)}")
    print(f"  Disfluency occurrences: {len(rows)}")
    print(f"  Audio clips created:    {clips_created}")
    if len(result_df) > 0:
        print(f"\n  By type:")
        print(result_df["disfluency_type"].value_counts().to_string(header=False))
    print(f"\n  Sheet saved to: {out_csv}")

    # Write methodology summary
    methodology = out_dir / "methodology.txt"
    methodology.write_text(
        "Disfluency Detection Methodology\n"
        "=================================\n\n"
        "1. How disfluencies were detected:\n"
        "   - Text-based analysis on segment-level transcriptions from the dataset.\n"
        "   - Fillers: matched against a curated list of Hindi filler words\n"
        "     (e.g., अ, अम्म, उह, हम्म, मतलब, तो, बस).\n"
        "   - Repetitions: detected consecutive identical words using token comparison.\n"
        "   - False starts: identified via hyphen/ellipsis patterns indicating interrupted speech.\n"
        "   - Prolongations: regex matching for repeated characters (e.g., हम्म्म).\n"
        "   - Hesitations: short segments (<= 3 tokens) composed entirely of backchannel words.\n\n"
        "2. How audio segments were clipped:\n"
        "   - Used the start/end timestamps from the transcription JSON for each segment.\n"
        "   - Clipped using ffmpeg with mono 16kHz output for consistency.\n"
        "   - Command: ffmpeg -ss <start> -to <end> -i <full_audio> -ac 1 -ar 16000 <clip>\n\n"
        "3. Preprocessing/Normalization:\n"
        "   - Unicode NFC normalization on all text.\n"
        "   - Whitespace collapsing and stripping.\n"
        "   - Punctuation-insensitive token matching for filler detection.\n",
        encoding="utf-8",
    )
    print(f"  Methodology saved to: {methodology}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q2: Disfluency detection and segmentation")
    ap.add_argument("--manifest", required=True, help="Segment-level manifest CSV from Q1")
    ap.add_argument("--audio_dir", required=True, help="Directory with downloaded audio files")
    ap.add_argument("--out_dir", required=True, help="Output directory for clips + sheet")
    main(ap.parse_args())
