# Q3: Spelling Classification Approach

## Result
- **Total unique words**: 7,448
- **Correct spelling**: 2,622 (35.2%)
- **Incorrect spelling**: 4,826 (64.8%)

## Approach: Multi-Signal Classification

No single Hindi spell-checker is comprehensive enough for conversational speech, so I used a multi-signal strategy:

### Signal 1: Known-Good Dictionary (~300+ words)
Manually curated list of common Hindi words that are unambiguously correct (pronouns, postpositions, common verbs, numbers, etc.). Any word in this set → correct.

### Signal 2: Devanagari Structure Validation
Checks whether the character sequence is structurally valid in Devanagari:
- No orphaned matras (vowel signs without preceding consonants)
- Valid conjunct formations
- Proper use of halant, nukta, anusvara
- Words failing structural checks → incorrect

### Signal 3: Frequency + Recording Spread
Words appearing across many different recordings (high "spread") are likely genuine vocabulary, not typos:
- Word in 5+ recordings → likely correct
- Word in only 1 recording with low frequency → likely typo/error

### Signal 4: English-in-Devanagari Handling
Per the guidelines, English words transcribed in Devanagari (e.g., "कंप्यूटर" for "computer") are correct. Detected using common transliteration patterns and suffixes.

## Why not a spell-checker API?
- Hindi spell-checkers have limited conversational vocabulary coverage
- They flag valid transliterated English words as errors
- Corpus-based signals leverage the dataset's own redundancy
- More interpretable and auditable than a black-box API
