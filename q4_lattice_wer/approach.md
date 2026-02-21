# Q4: Lattice-Based Fair WER — Approach

## Problem
When the human reference contains errors, models that transcribed correctly get unfairly penalized by WER. We need a method to construct a "fairer" reference.

## Alignment Unit: WORD

**Justification:**
- Hindi is space-delimited — word boundaries are unambiguous
- WER is a word-level metric, so matching the alignment unit avoids granularity mismatch
- Subword alignment adds complexity (how to split Hindi words into subwords?) without clear benefit
- Phrase-level loses granularity — can't pinpoint specific word errors

## Approach: ROVER-Style Confusion Network with Majority Voting

### Algorithm (Pseudocode):

```
For each utterance:
    1. ref_tokens = tokenize(human_reference)
    2. For each model_output in [Model H, Model i, ..., Model n]:
         alignment = levenshtein_align(ref_tokens, tokenize(model_output))
         # Returns pairs: (ref_word, hyp_word), (ref_word, None), (None, hyp_word)

    3. Build confusion network:
         For each reference position i:
             votes[i] = Counter of words each model produced at this position

    4. Construct pseudo-reference by majority voting:
         For each position i:
             top_word, count = most_voted_word(votes[i])
             if top_word != ref_tokens[i] AND count >= threshold:
                 pseudo_ref[i] = top_word   # Models agree reference is wrong
             else:
                 pseudo_ref[i] = ref_tokens[i]  # Keep reference

    5. Compute WER for each model against both:
         - Original human reference
         - Lattice pseudo-reference
```

### Agreement Threshold
Set to **3 out of 6 models** (50%). If 3+ models agree on a word different from the reference, the reference is likely wrong.

## Results

| Model | WER vs Reference | WER vs Lattice | Delta | Improved? |
|-------|-----------------|----------------|-------|-----------|
| Model H | 3.98% | 3.09% | -0.89 | Yes |
| Model i | 6.70% | 7.76% | +1.06 | No |
| Model n | 11.39% | 9.62% | -1.77 | Yes |
| Model l | 11.32% | 10.20% | -1.12 | Yes |
| Model m | 20.73% | 18.76% | -1.97 | Yes |
| Model k | 24.27% | 23.93% | -0.34 | Yes |

- **30.4% of references** were corrected by the lattice
- **5/6 models improved** — confirming the method reduces unfair penalties
- **Model i** got slightly worse, meaning it was "accidentally" matching some reference errors
- The method correctly distinguishes genuine model errors from reference errors
