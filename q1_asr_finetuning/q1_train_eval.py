#!/usr/bin/env python
"""
Q1 pipeline: preprocess Hindi ASR dataset, fine-tune whisper-small, and evaluate WER.

Usage:
  # Step 1: Download data from GCS
  python q1_train_eval.py download --input_csv ft_data_preprocessed_with_text.csv --out_dir data_q1

  # Step 2: Preprocess into segment-level manifests
  python q1_train_eval.py preprocess --input_csv ft_data_preprocessed_with_text.csv --out_dir data_q1

  # Step 3: Fine-tune whisper-small
  python q1_train_eval.py train --manifest_dir data_q1 --output_dir whisper_hi_finetuned

  # Step 4: Evaluate pretrained baseline on FLEURS
  python q1_train_eval.py eval --model_id openai/whisper-small --split test

  # Step 5: Evaluate fine-tuned model on FLEURS
  python q1_train_eval.py eval --model_id whisper_hi_finetuned --split test

  # Step 6: Compare both in a WER table
  python q1_train_eval.py compare --pretrained_id openai/whisper-small --finetuned_id whisper_hi_finetuned
"""

from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_hi_text(text: str) -> str:
    """Normalize Hindi text: NFC, remove ZWJ/ZWNJ, collapse whitespace."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    # Remove REDACTED tokens (present in dataset)
    text = re.sub(r"\bREDACTED\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def resolve_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure rec_url, transcription_url, metadata_url columns exist.

    The actual CSV has columns named rec_url_new / transcription_url_new /
    metadata_url_new.  We normalise them here so downstream code uses a
    single naming convention.
    """
    out = df.copy()

    # Audio URL
    if "rec_url_new" in out.columns:
        out["rec_url"] = out["rec_url_new"]
    elif "rec_url_gcp" in out.columns:
        out["rec_url"] = out["rec_url_gcp"]
    elif "rec_url" not in out.columns:
        out["rec_url"] = out.apply(
            lambda r: f"https://storage.googleapis.com/upload_goai/{int(r.user_id)}/{int(r.recording_id)}_audio.wav",
            axis=1,
        )

    # Transcription URL
    if "transcription_url_new" in out.columns:
        out["transcription_url"] = out["transcription_url_new"]
    elif "transcription_url" not in out.columns:
        out["transcription_url"] = out.apply(
            lambda r: f"https://storage.googleapis.com/upload_goai/{int(r.user_id)}/{int(r.recording_id)}_transcription.json",
            axis=1,
        )

    # Metadata URL
    if "metadata_url_new" in out.columns:
        out["metadata_url"] = out["metadata_url_new"]
    elif "metadata_url" not in out.columns:
        out["metadata_url"] = out.apply(
            lambda r: f"https://storage.googleapis.com/upload_goai/{int(r.user_id)}/{int(r.recording_id)}_metadata.json",
            axis=1,
        )

    return out


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_one(url: str, dest: Path) -> bool:
    """Download a single URL to dest.  Returns True on success."""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"  FAILED {url}: {e}")
        return False


def download(args: argparse.Namespace) -> None:
    """Download audio + transcription files from GCS URLs."""
    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    trans_dir = out_dir / "transcriptions"
    audio_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = resolve_urls(df)
    df = df[df["language"].astype(str).str.lower().eq("hi")].copy()

    tasks = []
    for _, r in df.iterrows():
        rec_id = int(r["recording_id"])
        tasks.append((r["rec_url"], audio_dir / f"{rec_id}.wav"))
        tasks.append((r["transcription_url"], trans_dir / f"{rec_id}_transcription.json"))

    print(f"Downloading {len(tasks)} files ({len(df)} recordings x 2) ...")

    success, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, url, dst): (url, dst) for url, dst in tasks}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            if fut.result():
                success += 1
            else:
                fail += 1
            if i % 20 == 0 or i == len(futures):
                print(f"  [{i}/{len(futures)}] success={success} fail={fail}")

    print(f"Download complete: {success} ok, {fail} failed")


# ---------------------------------------------------------------------------
# Transcription JSON parsing
# ---------------------------------------------------------------------------

def _safe_json_load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_segments_from_transcription(payload: Any) -> List[Dict[str, Any]]:
    """Parse segment list from transcription JSON.

    The Josh Talks transcription format is a top-level JSON array:
      [ {"start": 37.54, "end": 42.7, "speaker_id": ..., "text": "..."}, ... ]

    Also handles dict formats with keys like segments/utterances/results/chunks.
    """
    # Handle top-level array (the actual format used by this dataset)
    if isinstance(payload, list):
        segments_list = payload
    elif isinstance(payload, dict):
        candidates = [
            payload.get("segments"),
            payload.get("utterances"),
            payload.get("results"),
            payload.get("chunks"),
        ]
        segments_list = None
        for segs in candidates:
            if isinstance(segs, list) and segs:
                segments_list = segs
                break
        if segments_list is None:
            txt = payload.get("text") or payload.get("transcript")
            if isinstance(txt, str) and txt.strip():
                return [{"segment_id": 0, "start": None, "end": None, "text": normalize_hi_text(txt)}]
            return []
    else:
        return []

    parsed = []
    for i, s in enumerate(segments_list):
        if not isinstance(s, dict):
            continue
        start = s.get("start") or s.get("start_time") or s.get("from") or s.get("begin")
        end = s.get("end") or s.get("end_time") or s.get("to") or s.get("finish")
        text = s.get("text") or s.get("transcript") or s.get("utterance") or s.get("sentence")
        if text is None:
            continue
        text = normalize_hi_text(text)
        if not text:
            continue
        try:
            start_f = float(start) if start is not None else None
            end_f = float(end) if end is not None else None
        except Exception:
            start_f, end_f = None, None
        parsed.append({"segment_id": i, "start": start_f, "end": end_f, "text": text})
    return parsed


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------

def preprocess(args: argparse.Namespace) -> None:
    """Build segment-level manifests from the CSV + downloaded transcription JSONs."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = resolve_urls(df)
    df = df[df["language"].astype(str).str.lower().eq("hi")].copy()

    rows = []
    skipped_no_json = 0
    for _, r in df.iterrows():
        rec_id = int(r["recording_id"])
        user_id = int(r["user_id"])
        split = str(r.get("split", "train"))
        audio_path = str(out_dir / "audio" / f"{rec_id}.wav")

        # Try loading the downloaded transcription JSON (has segments with timestamps)
        local_transcript = out_dir / "transcriptions" / f"{rec_id}_transcription.json"
        payload = _safe_json_load(local_transcript)

        if payload:
            segments = parse_segments_from_transcription(payload)
        else:
            # Fallback: use the full text from CSV (no timestamp info)
            skipped_no_json += 1
            full_text = normalize_hi_text(str(r.get("text", "")))
            segments = [{"segment_id": 0, "start": None, "end": None, "text": full_text}] if full_text else []

        for s in segments:
            text = s["text"]
            if not text:
                continue
            rows.append({
                "user_id": user_id,
                "recording_id": rec_id,
                "split": split,
                "audio_path": audio_path,
                "segment_id": s["segment_id"],
                "start": s["start"],
                "end": s["end"],
                "text": text,
            })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest_all.csv", index=False)
    for sp in sorted(manifest["split"].unique()):
        manifest[manifest["split"] == sp].to_csv(out_dir / f"manifest_{sp}.csv", index=False)

    if skipped_no_json:
        print(f"WARNING: {skipped_no_json} recordings had no transcription JSON (used CSV text fallback)")

    print(f"\nPreprocess Summary:")
    print(f"  Total segments: {len(manifest)}")
    print(f"  Split distribution: {manifest['split'].value_counts().to_dict()}")
    print(f"  Manifests saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Train (fine-tune Whisper-small)
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    try:
        from dataclasses import dataclass
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
        import evaluate
        import numpy as np
        import torch
        import librosa
    except ImportError as e:
        raise SystemExit(
            "Missing dependencies. Install:\n"
            "  pip install transformers datasets evaluate librosa soundfile accelerate jiwer"
        ) from e

    manifest_dir = Path(args.manifest_dir)
    train_df = pd.read_csv(manifest_dir / "manifest_train.csv")
    val_df = pd.read_csv(manifest_dir / "manifest_validation.csv")

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="Hindi", task="transcribe")
    model.generation_config.suppress_tokens = []

    # --- Data collator for Whisper (handles padding of input_features + labels) ---
    @dataclass
    class DataCollatorSpeechSeq2Seq:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding token id with -100 so they are ignored in loss
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            # Remove the decoder_start_token if it was prepended
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --- Lazy-loading PyTorch Dataset (avoids caching all audio in RAM) ---
    class WhisperASRDataset(torch.utils.data.Dataset):
        def __init__(self, df: pd.DataFrame, processor, max_label_len: int = 448):
            self.df = df.reset_index(drop=True)
            self.processor = processor
            self.max_label_len = max_label_len

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            audio_path = row["audio_path"]
            start = row.get("start")
            end = row.get("end")

            # Load only the segment we need using offset/duration
            offset = 0.0
            duration = None
            if pd.notna(start) and pd.notna(end):
                try:
                    offset = float(start)
                    duration = float(end) - offset
                    if duration <= 0:
                        offset, duration = 0.0, None
                except (ValueError, TypeError):
                    offset, duration = 0.0, None

            y, _ = librosa.load(audio_path, sr=16000, mono=True,
                                offset=offset, duration=duration)

            if len(y) == 0:
                y = np.zeros(16000, dtype=np.float32)

            feats = self.processor.feature_extractor(
                y, sampling_rate=16000
            ).input_features[0]
            labels = self.processor.tokenizer(
                row["text"],
                truncation=True,
                max_length=self.max_label_len,
            ).input_ids

            return {"input_features": feats, "labels": labels}

    print("Building lazy-loading datasets ...")
    train_ds = WhisperASRDataset(train_df, processor)
    val_ds = WhisperASRDataset(val_df, processor)

    # --- Metrics ---
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    # --- Training arguments (compatible with transformers v4 and v5) ---
    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    # --- Trainer (compatible with transformers v4 and v5) ---
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = processor.feature_extractor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("Starting training ...")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nSaved fine-tuned model to {args.output_dir}")


# ---------------------------------------------------------------------------
# Evaluate on FLEURS
# ---------------------------------------------------------------------------

def eval_model(args: argparse.Namespace) -> None:
    try:
        import evaluate
        from datasets import load_dataset
        from transformers import pipeline
    except ImportError as e:
        raise SystemExit(
            "Missing dependencies. Install:\n"
            "  pip install transformers datasets evaluate jiwer"
        ) from e

    print(f"Loading model: {args.model_id} ...")
    asr = pipeline(
        task="automatic-speech-recognition",
        model=args.model_id,
        device=0 if args.device == "cuda" else -1,
        generate_kwargs={"language": "hindi", "task": "transcribe"},
    )

    print(f"Loading FLEURS hi_in split={args.split} ...")
    ds = load_dataset("google/fleurs", "hi_in", split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Evaluating {len(ds)} samples ...")
    refs, preds = [], []
    for i, row in enumerate(ds):
        refs.append(normalize_hi_text(row["transcription"]))
        pred = asr(row["audio"]["array"], chunk_length_s=30, batch_size=8)
        preds.append(normalize_hi_text(pred["text"]))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}]")

    wer_metric = evaluate.load("wer")
    score = wer_metric.compute(references=refs, predictions=preds)

    print(f"\n{'='*50}")
    print(f"Model:  {args.model_id}")
    print(f"Split:  {args.split}")
    print(f"Samples: {len(ds)}")
    print(f"WER:    {score*100:.2f}%")
    print(f"{'='*50}")

    # Save result to JSON for later comparison
    result = {"model_id": args.model_id, "split": args.split, "samples": len(ds), "wer": score}
    out_path = Path(args.model_id.replace("/", "_") + f"_wer_{args.split}.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Result saved to {out_path}")


# ---------------------------------------------------------------------------
# Compare: side-by-side WER table
# ---------------------------------------------------------------------------

def compare(args: argparse.Namespace) -> None:
    """Evaluate pretrained and fine-tuned models, print WER comparison table."""
    try:
        import evaluate
        from datasets import load_dataset
        from transformers import pipeline
    except ImportError as e:
        raise SystemExit(
            "Missing dependencies. Install:\n"
            "  pip install transformers datasets evaluate jiwer"
        ) from e

    print(f"Loading FLEURS hi_in split={args.split} ...")
    ds = load_dataset("google/fleurs", "hi_in", split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    models = {
        "Whisper-small (pretrained)": args.pretrained_id,
        "Whisper-small (fine-tuned)": args.finetuned_id,
    }

    results = {}
    wer_metric = evaluate.load("wer")

    for label, model_id in models.items():
        print(f"\nEvaluating: {label} ({model_id}) ...")
        asr = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=0 if args.device == "cuda" else -1,
            generate_kwargs={"language": "hindi", "task": "transcribe"},
        )

        refs, preds = [], []
        for i, row in enumerate(ds):
            refs.append(normalize_hi_text(row["transcription"]))
            pred = asr(row["audio"]["array"], chunk_length_s=30, batch_size=8)
            preds.append(normalize_hi_text(pred["text"]))
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(ds)}]")

        score = wer_metric.compute(references=refs, predictions=preds)
        results[label] = score

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"WER Comparison on FLEURS Hindi ({args.split}, {len(ds)} samples)")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'WER (%)':<10}")
    print(f"{'-'*35} {'-'*10}")
    for label, score in results.items():
        print(f"{label:<35} {score*100:.2f}%")
    print(f"{'='*60}")

    # Save to CSV
    rows = [{"model": k, "wer_pct": round(v * 100, 2)} for k, v in results.items()]
    out_csv = Path("wer_comparison.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nTable saved to {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Q1: Hindi ASR pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download audio + transcription files from GCS")
    p_dl.add_argument("--input_csv", required=True, help="Path to ft_data_preprocessed_with_text.csv")
    p_dl.add_argument("--out_dir", required=True, help="Output directory for downloaded files")
    p_dl.add_argument("--workers", type=int, default=8, help="Parallel download threads")
    p_dl.set_defaults(func=download)

    # preprocess
    p_pre = sub.add_parser("preprocess", help="Build segment-level manifests")
    p_pre.add_argument("--input_csv", required=True)
    p_pre.add_argument("--out_dir", required=True)
    p_pre.set_defaults(func=preprocess)

    # train
    p_train = sub.add_parser("train", help="Fine-tune whisper-small on Hindi data")
    p_train.add_argument("--manifest_dir", required=True)
    p_train.add_argument("--output_dir", required=True)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--learning_rate", type=float, default=1e-5)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    p_train.set_defaults(func=train)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a model on FLEURS Hindi test")
    p_eval.add_argument("--model_id", required=True)
    p_eval.add_argument("--split", default="test")
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p_eval.set_defaults(func=eval_model)

    # compare
    p_cmp = sub.add_parser("compare", help="Side-by-side WER comparison table")
    p_cmp.add_argument("--pretrained_id", default="openai/whisper-small")
    p_cmp.add_argument("--finetuned_id", required=True)
    p_cmp.add_argument("--split", default="test")
    p_cmp.add_argument("--limit", type=int, default=0)
    p_cmp.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p_cmp.set_defaults(func=compare)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
