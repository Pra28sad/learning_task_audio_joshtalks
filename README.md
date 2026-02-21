# Task Assignment | AI Researcher Intern — Speech & Audio | Josh Talks

## Results at a Glance

| Question | Task | Key Result |
|----------|------|-----------|
| **Q1** | Hindi ASR Fine-tuning | WER: 61.78% → **36.66%** (40.7% relative improvement) |
| **Q2** | Disfluency Detection | **7,724** occurrences detected, **3,992** audio clips |
| **Q3** | Spelling Classification | 7,448 unique words → **2,622 correct**, **4,826 incorrect** |
| **Q4** | Lattice-based Fair WER | **30.4%** of references corrected, **5/6 models** improved |

## Full Report

See **[submission_report.ipynb](submission_report.ipynb)** for the complete report with visualizations, methodology explanations, and analysis.

## Repository Structure

```
submission/
│
├── README.md                              ← You are here
├── submission_report.ipynb                ← Full report with charts (run this)
├── submission_report_executed.ipynb        ← Pre-executed version with outputs
│
├── q1_asr_finetuning/
│   ├── q1_train_eval.py                   ← Code: download, preprocess, train, evaluate
│   ├── wer_results.csv                    ← WER table (in requested format)
│   └── preprocessing_approach.md          ← Preprocessing & training methodology
│
├── q2_disfluency_detection/
│   ├── q2_disfluency_extra.py             ← Code: disfluency detection + audio clipping
│   ├── disfluency_sheet.csv               ← Output: 7,724 rows (in requested format)
│   └── methodology.txt                    ← Detection methodology summary
│
├── q3_spelling_classification/
│   ├── q3_spelling_label.py               ← Code: multi-signal spelling classifier
│   ├── spelling_classification.csv        ← Output: 7,448 words (word + label, 2 columns)
│   └── approach.md                        ← Classification approach & reasoning
│
├── q4_lattice_wer/
│   ├── q4_lattice_wer.py                  ← Code: ROVER-style lattice WER
│   ├── q4_results.csv                     ← Output: WER comparison per model
│   └── approach.md                        ← Lattice design + alignment justification
│
└── figures/
    ├── fig_q1_wer_comparison.png
    ├── fig_q2_disfluency_dist.png
    ├── fig_q3_spelling_dist.png
    └── fig_q4_lattice_wer.png
```

## Q2 Audio Clips

The 3,992 segmented disfluency audio clips (910 MB) are available via Google Drive:
> **[Link to be added after upload]**

These clips correspond to each row in `q2_disfluency_detection/disfluency_sheet.csv`.

## How to Reproduce

```bash
# Environment setup
python -m venv venv && source venv/bin/activate
pip install torch==2.6.0+cu124 -f https://download.pytorch.org/whl/cu124
pip install transformers datasets==3.6.0 evaluate jiwer librosa soundfile pandas matplotlib

# Q1: Hindi ASR Pipeline (download → preprocess → train → evaluate)
python q1_asr_finetuning/q1_train_eval.py download \
    --input_csv ft_data_preprocessed_with_text.csv --out_dir data_q1
python q1_asr_finetuning/q1_train_eval.py preprocess \
    --input_csv ft_data_preprocessed_with_text.csv --out_dir data_q1
python q1_asr_finetuning/q1_train_eval.py train \
    --manifest_dir data_q1 --output_dir whisper_hi_finetuned --epochs 3
python q1_asr_finetuning/q1_train_eval.py compare \
    --pretrained_id openai/whisper-small --finetuned_id whisper_hi_finetuned --split test

# Q2: Disfluency Detection
python q2_disfluency_detection/q2_disfluency_extra.py \
    --manifest data_q1/manifest_all.csv --audio_dir data_q1/audio \
    --out_dir data_q2_disfluency

# Q3: Spelling Classification
python q3_spelling_classification/q3_spelling_label.py \
    --manifest data_q1/manifest_all.csv --out_csv spelling_results.csv

# Q4: Lattice WER
python q4_lattice_wer/q4_lattice_wer.py \
    --input_csv q4_data.csv --out_csv q4_results.csv
```

## Requirements
- Python 3.10+
- CUDA-capable GPU (for Q1 training/evaluation)
- ~8 GB GPU memory
- ~10 GB disk for audio downloads
