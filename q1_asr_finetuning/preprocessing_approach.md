# Q1: Preprocessing & Fine-tuning Approach

## a) Data Preprocessing

### Steps:
1. **Download**: Fetched 104 audio files (.wav) and transcription JSONs from GCS using the URL pattern:
   `https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_audio.wav`

2. **Parse transcriptions**: Each transcription JSON is a top-level array of segments:
   `[{"start": 0.0, "end": 5.2, "speaker_id": "S1", "text": "..."}, ...]`

3. **Segment-level splitting**: Split each recording into segments using the JSON timestamps.
   This produced **5,732 segments** from 104 recordings (~22 hours of audio).
   - Train: 4,801 segments (85 recordings)
   - Validation: 375 segments (6 recordings)
   - Test: 556 segments (13 recordings)

4. **Text normalization**: Unicode NFC normalization, whitespace collapsing, stripped punctuation for WER evaluation.

5. **Audio loading**: Used `librosa.load(path, sr=16000, offset=start, duration=end-start)` to load only the segment we need, avoiding loading full recordings into memory.

### Why segment-level, not full recordings?
- Whisper's input limit is 30 seconds
- Segment-level gives more training examples
- Aligns text with the exact audio portion
- Avoids Whisper's own chunking which can cause errors at boundaries

## b) Fine-tuning Configuration

| Parameter | Value |
|-----------|-------|
| Base model | openai/whisper-small (244M params) |
| Language | Hindi |
| Epochs | 3 |
| Batch size | 2 (effective: 8 with grad_accum=4) |
| Learning rate | 1e-5 |
| Optimizer | AdamW |
| Training time | ~47 minutes (single GPU) |

### Training Progression:
| Epoch | Train Loss | Eval Loss | Eval WER |
|-------|-----------|-----------|----------|
| 1     | 0.7116    | 0.4044    | 43.14%   |
| 2     | 0.2415    | 0.3693    | 37.50%   |
| 3     | 0.0956    | 0.3638    | 35.95%   |

## c) WER Results (FLEURS Hindi Test, 418 samples)

| Model | WER |
|-------|-----|
| Whisper Small (Pretrained) | 0.6178 (61.78%) |
| FT Whisper Small (ours) | 0.3666 (36.66%) |

**Improvement: 25.12 percentage points (40.7% relative reduction)**
