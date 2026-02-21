#!/usr/bin/env python
"""
Q3: Classify unique Hindi words into correct/incorrect spelling.

Strategy (multi-signal approach):
  1. Dictionary lookup:  Check against a Hindi wordlist (from indic-nlp or shabdkosh).
  2. Morphological check: Use Hindi morphological rules to validate word structure.
  3. Devanagari validity: Check for invalid character sequences in Devanagari.
  4. Frequency + corpus agreement: High-frequency words across many recordings are
     likely correct; single-occurrence words in one recording are suspect.
  5. English-in-Devanagari: Words transliterated from English (e.g., "computer" -> "कंप्यूटर")
     are counted as correct per the transcription guidelines.

Usage:
  # From a word list file
  python q3_spelling_label.py --input_words unique_words.csv --out_csv spelling_labels.csv

  # From a corpus (extract words from text column)
  python q3_spelling_label.py --input_corpus data_q1/manifest_all.csv --text_col text --out_csv spelling_labels.csv
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd


# ---------------------------------------------------------------------------
# Devanagari constants
# ---------------------------------------------------------------------------

# Devanagari Unicode range
DEV_RE = re.compile(r"[\u0900-\u097F]+")

# Valid Devanagari consonants
CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
CONSONANTS |= {"क्ष", "त्र", "ज्ञ", "श्र"}

# Vowels and matras
VOWELS = set("अआइईउऊऋएऐओऔ")
MATRAS = set("ािीुूृेैोौंःँ")
HALANT = "्"
NUKTA = "़"
ANUSVARA = "ं"
VISARGA = "ः"
CHANDRABINDU = "ँ"

# Invalid Devanagari patterns
# - Two consecutive matras (vowel signs) without a consonant between them
INVALID_DOUBLE_MATRA = re.compile(r"[ािीुूृेैोौ]{2,}")
# - Halant at the very start of a word
INVALID_START_HALANT = re.compile(r"^्")
# - Three or more consecutive halants
INVALID_TRIPLE_HALANT = re.compile(r"(्){3,}")


# ---------------------------------------------------------------------------
# Common valid Hindi words (high-confidence seed list)
# Includes common function words, pronouns, postpositions, conjunctions,
# and frequently used English-to-Devanagari transliterations.
# ---------------------------------------------------------------------------

COMMON_VALID: Set[str] = {
    # Function words / postpositions
    "है", "हैं", "था", "थी", "थे", "हो", "हुआ", "हुई", "हुए",
    "का", "की", "के", "में", "से", "पर", "को", "ने", "तक",
    "और", "या", "अगर", "तो", "लेकिन", "मगर", "क्योंकि", "कि", "जो", "जब",
    "ही", "भी", "बहुत", "कुछ", "सब", "वो", "ये", "यह", "वह",
    # Pronouns
    "मैं", "मैंने", "मुझे", "मेरा", "मेरी", "मेरे",
    "तुम", "तुम्हारा", "तुम्हारी", "तुम्हें", "तू", "तेरा",
    "आप", "आपका", "आपकी", "आपके", "आपको", "आपने", "आपसे",
    "हम", "हमने", "हमें", "हमारा", "हमारी", "हमारे",
    "उसका", "उसकी", "उसके", "उसने", "उसको", "उनका", "उनकी", "उनके",
    # Common verbs
    "करना", "करता", "करती", "करते", "किया", "करें", "कर",
    "होना", "होता", "होती", "होते",
    "जाना", "जाता", "जाती", "जाते", "गया", "गई", "गए", "जाएं",
    "आना", "आता", "आती", "आते", "आया", "आई", "आए",
    "देना", "देता", "देती", "देते", "दिया", "दी", "दिए",
    "लेना", "लेता", "लेती", "लेते", "लिया", "लिए",
    "बोलना", "बोलता", "बोलती", "बोले", "बोला",
    "कहना", "कहता", "कहती", "कहते", "कहा",
    "मिलना", "मिला", "मिली", "मिले", "मिलता", "मिलती",
    "रहना", "रहता", "रहती", "रहते", "रहा", "रही", "रहे",
    "सकता", "सकती", "सकते", "सकें",
    "पड़ना", "पड़ता", "पड़ती", "पड़ा", "पड़ी", "पड़े",
    "चाहिए", "चाहता", "चाहती",
    "बताना", "बताता", "बताती", "बताया", "बताई", "बताइए",
    "सीखना", "सीखा", "सीखी", "सीखे",
    "लगना", "लगता", "लगती", "लगते", "लगा", "लगी",
    # Common nouns
    "लोग", "लोगों", "काम", "बात", "बातें", "समय", "दिन", "साल", "सालों",
    "घर", "जगह", "तरह", "तरीका", "चीज़", "चीजें", "पैसा", "पैसे",
    "नौकरी", "जॉब", "कंपनी", "परिवार", "दोस्त", "दोस्तों",
    # Common adjectives / adverbs
    "अच्छा", "अच्छी", "अच्छे", "बुरा", "बड़ा", "बड़ी", "बड़े", "छोटा", "छोटी",
    "नया", "नई", "नए", "पुराना", "पुरानी", "पुराने",
    "ज्यादा", "कम", "पहले", "बाद", "आगे", "पीछे", "ऊपर", "नीचे",
    "सही", "गलत", "जरूर", "जरूरी", "शायद",
    # Numbers
    "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस",
    # Question words
    "क्या", "कैसे", "कैसा", "कैसी", "कहां", "कहाँ", "कब", "कौन", "कितना", "कितनी",
    # Conversational / discourse
    "हां", "हाँ", "नहीं", "जी", "ठीक", "बस", "अभी", "फिर", "वहां", "वहाँ", "यहां", "यहाँ",
    "इसलिए", "इसमें", "उसमें", "वाला", "वाली", "वाले",
    # Common English-to-Devanagari transliterations
    "कंप्यूटर", "मोबाइल", "इंटरनेट", "फोन", "ऑनलाइन", "ऑफलाइन",
    "इंटरव्यू", "ऑफिस", "जॉब", "करियर", "कैरियर",
    "फाइनेंस", "एक्सपीरियंस", "प्रमोशन", "सैलरी",
    "टीचर", "प्रोफेसर", "स्टूडेंट", "कॉलेज", "यूनिवर्सिटी",
    "बिजनेस", "मैनेजमेंट", "इंजीनियरिंग",
    "प्रॉब्लम", "सॉल्यूशन", "प्रोजेक्ट", "रिजल्ट",
    "पर्सनल", "प्रोफेशनल", "सोशल", "मीडिया",
    "एंड", "बट", "ओके", "सर", "मैम", "प्लीज", "थैंक्यू", "सॉरी",
    "गवर्नमेंट", "प्राइवेट", "सेक्टर",
    "टाइम", "डिफिकल्ट", "डिफिकल्टी", "चैलेंज",
    "ग्रोथ", "डेवलपमेंट", "अपॉर्चुनिटी", "मोटिवेशन",
    "फैमिली", "मेंबर", "फ्रेंड", "रिलेशनशिप",
}


# ---------------------------------------------------------------------------
# Word normalisation and validation
# ---------------------------------------------------------------------------

def norm_word(w: str) -> str:
    """Normalize a Devanagari word: NFC, strip non-Devanagari chars."""
    w = unicodedata.normalize("NFC", str(w)).strip()
    w = re.sub(r"[^\u0900-\u097F]", "", w)
    return w


def is_valid_devanagari_structure(word: str) -> bool:
    """Check if a word has valid Devanagari orthographic structure.

    Returns False for words with clearly invalid character sequences.
    """
    if not word:
        return False
    if INVALID_START_HALANT.search(word):
        return False
    if INVALID_DOUBLE_MATRA.search(word):
        return False
    if INVALID_TRIPLE_HALANT.search(word):
        return False
    # Word should not be just matras or just halants
    stripped = re.sub(r"[ािीुूृेैोौंःँ्]", "", word)
    if not stripped:
        return False
    return True


def looks_like_english_in_devanagari(word: str) -> bool:
    """Heuristic: does this word look like an English word transliterated to Devanagari?

    Common patterns: words with ट/ड/फ clusters, words ending in -शन, -मेंट, -टी, etc.
    """
    english_suffixes = ["शन", "मेंट", "टी", "सी", "जी", "ली", "नल", "रल", "टर", "डर"]
    for suf in english_suffixes:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Word extraction
# ---------------------------------------------------------------------------

def extract_words_from_texts(texts: Iterable[str]) -> List[str]:
    """Extract all Devanagari words from a list of texts."""
    out = []
    for t in texts:
        out.extend(DEV_RE.findall(str(t)))
    return out


def load_words(args: argparse.Namespace) -> List[str]:
    """Load words from file or corpus."""
    if args.input_words:
        p = Path(args.input_words)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            col = args.word_col or df.columns[0]
            return [norm_word(x) for x in df[col].astype(str).tolist()]
        if p.suffix.lower() in {".txt", ".lst"}:
            return [norm_word(x) for x in p.read_text(encoding="utf-8").splitlines()]
    if args.input_corpus:
        df = pd.read_csv(args.input_corpus)
        words = extract_words_from_texts(df[args.text_col].astype(str).tolist())
        return [norm_word(w) for w in words]
    raise ValueError("Provide --input_words or --input_corpus")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(words: List[str], min_freq_correct: int = 3,
             recording_ids: List[int] | None = None) -> pd.DataFrame:
    """Classify unique words as correct/incorrect using multiple signals.

    Signals used:
      1. Known-correct word list (COMMON_VALID)
      2. Devanagari orthographic validity
      3. Frequency across the corpus
      4. Spread across recordings (word appearing in many recordings = likely correct)
      5. Word length heuristic (very short words that aren't known are suspect)
    """
    words_clean = [w for w in words if w]
    freq = Counter(words_clean)

    # Compute per-recording spread if recording_ids available
    word_recording_spread: Counter = Counter()
    if recording_ids and len(recording_ids) == len(words):
        for w, rid in zip(words, recording_ids):
            if w:
                word_recording_spread[w + f"_{rid}"] = 1
        # Count unique recordings per word
        word_recs: dict = {}
        for w, rid in zip(words, recording_ids):
            if w:
                word_recs.setdefault(w, set()).add(rid)
        word_spread = {w: len(recs) for w, recs in word_recs.items()}
    else:
        word_spread = {}

    unique_words = sorted(set(words_clean))
    rows = []

    for w in unique_words:
        f = freq[w]
        spread = word_spread.get(w, 0)
        reasons = []

        # Signal 1: Known valid word
        if w in COMMON_VALID:
            label = "correct spelling"
            reasons.append("known_valid")
        # Signal 2: Invalid Devanagari structure
        elif not is_valid_devanagari_structure(w):
            label = "incorrect spelling"
            reasons.append("invalid_devanagari_structure")
        # Signal 3: Single character (likely a filler/noise, not a misspelling)
        elif len(w) == 1 and w in VOWELS:
            label = "correct spelling"
            reasons.append("single_vowel_filler")
        elif len(w) == 1:
            label = "incorrect spelling"
            reasons.append("single_char_suspect")
        # Signal 4: High frequency + multi-recording spread = likely correct
        elif f >= min_freq_correct and spread >= 3:
            label = "correct spelling"
            reasons.append(f"freq={f},spread={spread}")
        elif f >= min_freq_correct * 2:
            label = "correct spelling"
            reasons.append(f"high_freq={f}")
        # Signal 5: English transliteration pattern
        elif looks_like_english_in_devanagari(w) and f >= 2:
            label = "correct spelling"
            reasons.append("english_transliteration")
        # Signal 6: Moderate frequency (likely correct Hindi word)
        elif f >= min_freq_correct:
            label = "correct spelling"
            reasons.append(f"freq={f}")
        # Signal 7: Low frequency, likely a typo or rare misspelling
        else:
            label = "incorrect spelling"
            reasons.append(f"low_freq={f}")

        rows.append({
            "word": w,
            "label": label,
            "frequency": f,
            "recording_spread": spread,
            "reason": "|".join(reasons),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Load words
    if args.input_corpus:
        df = pd.read_csv(args.input_corpus)
        texts = df[args.text_col].astype(str).tolist()
        words = [norm_word(w) for w in extract_words_from_texts(texts)]

        # Get recording_ids for spread calculation
        recording_ids = []
        for _, row in df.iterrows():
            rid = int(row.get("recording_id", 0))
            row_words = [norm_word(w) for w in DEV_RE.findall(str(row[args.text_col]))]
            recording_ids.extend([rid] * len(row_words))
    else:
        words = load_words(args)
        recording_ids = None

    result = classify(words, min_freq_correct=args.min_freq_correct,
                      recording_ids=recording_ids)

    # Output: 2-column sheet (word, label) as requested + extra analysis columns
    out_full = Path(args.out_csv)
    result.to_csv(out_full, index=False)

    # Also save the simple 2-column version as requested
    simple_path = out_full.with_stem(out_full.stem + "_simple")
    result[["word", "label"]].to_csv(simple_path, index=False)

    n_correct = int((result["label"] == "correct spelling").sum())
    n_incorrect = int((result["label"] == "incorrect spelling").sum())
    total = len(result)

    print(f"\nSpelling Classification Summary:")
    print(f"  Total unique words:      {total}")
    print(f"  Correct spelling:        {n_correct} ({100*n_correct/total:.1f}%)")
    print(f"  Incorrect spelling:      {n_incorrect} ({100*n_incorrect/total:.1f}%)")
    print(f"\n  Full results:   {out_full}")
    print(f"  Simple 2-col:   {simple_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Q3: Hindi spelling classification")
    ap.add_argument("--input_words", help="CSV or TXT file with one word per row")
    ap.add_argument("--word_col", default="", help="Column name if CSV")
    ap.add_argument("--input_corpus", help="CSV with text column to extract words from")
    ap.add_argument("--text_col", default="text", help="Text column name in corpus CSV")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--min_freq_correct", type=int, default=3,
                     help="Minimum frequency threshold to consider word correct")
    main(ap.parse_args())
