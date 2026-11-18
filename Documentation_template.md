# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Hedonists  
**Team Members:** Harish Pal, Kush Agarwal, Suryansh Singh, Aryan Ghai
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
A blocking + classifier pipeline with an entity-aware decision stage. Unicode-safe normalisation
(Indian scripts mapped to English with a vocabulary learned from training matches) feeds an
11-key country-partitioned blocking index; a learned stage-2 ranker keeps the 40 most likely
candidates per Source-1 entity; a LightGBM matcher scores them with ~120 features, including
token-level *difference* features that separate deliberate distractor edits from corruption
noise and *competition* features that ask whether another reference entity fits a record better;
an out-of-sample stacker and a one-to-one, macro-F0.5-tuned threshold produce the matches.
Validation macro F0.5 rose from 0.9305 (first version) to **0.9816** (cross-fitted).

---

## 2. Methodology

### 2.1 Problem Analysis
Full-data EDA (`docs/EDA_REPORT.md`) and an audit of the first pipeline (`docs/EXPERIMENTS_V2.md`):
- 2.2M / 5.0M / 5.3M train records (S1/S2/S3); 1.73M / 4.89M / 5.08M test; test adds France (15% of S1).
- 5.58% of S1 are singletons; mean 3.46 matches; 100% of true pairs share the country label; no
  S2/S3 record belongs to two S1 (one-to-one constraint); ~26% of S2/S3 records are distractors.
- Noise: Indian-script names, accents, digit-for-letter substitutions, legal forms added / dropped,
  trade-name markers, URL/ID decorations, names replaced by pseudo-brands, empty addresses (~3%),
  truncated house numbers, abbreviations, state codes vs names.
- Distractors are near-copies of a real business with ONE deliberate change (house number changed,
  one name word substituted, Private -> Public), whereas true-match noise corrupts the same content
  (number truncated, word misspelt). 50.7% of S1 records share their exact name with another S1.

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned candidate ranker + classifier + entity-aware stacker + one-to-one assignment  
**Core Innovation:** (1) blocking without a shared-key-count pre-cap: every key-union candidate is
scored by a learned ranker (candidate recall 0.914 -> 0.991); (2) token-alignment difference features
(exact / fuzzy / substituted tokens, truncated vs changed numbers); (3) competition features over the
reference entities that share a record's name; (4) an out-of-sample stacker.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used** (all prefixed by country; blocks larger than 1000 records dropped):
  name skeleton word pairs, first name word, 5- and 8-char compact-name prefixes, full compact name
  (catches `visionphysicians.com`), house number + address word (incl. 1-digit numbers), state + first
  name skeleton, address words, address-word pairs (catches replaced names), name word x address word.
- Stage 1: union of all key blocks (pairs carry which keys fired and an IDF weight; a per-S1 safety
  cap keeps the highest-IDF pairs: 4000 in validation, 1200 in the test run for speed).
- Stage 2: a LightGBM ranker (trained on fit S1 only; `max_delta_step`/L2 needed at 0.1% positives)
  scores every stage-1 pair from cheap signals; the top 40 per S1 are the candidates
  (= `candidate_pairs.tsv`).
- **Blocking recall (60k validation S1):** 0.9908 (first version 0.9143); perfect-matcher ceiling 0.9970.
- **How true matches are protected:** keys on name AND address, phonetic skeletons, no count-based
  pre-cap (it lost 5.4% of true pairs and was biased against Source 3).

---

## 4. Matching Model

**Features used:**
- Name: ratio / token-set / token-sort / partial / Jaro-Winkler, skeleton and compact-name
  similarities, trade-name alias, legal forms; token alignment (exact, fuzzy = typo, substituted,
  frequency rank of substituted words), `public`/legal-form flags, pseudo-brand likelihood
  (character-trigram model learned on fit pairs).
- Address: token-set / ratio / partial / Jaccard, house numbers (exact, truncated, one-digit edit,
  unmatched, minimum numeric gap), state agreement, missing-address flags.
- Other: blocking keys fired + IDF, ranker score and rank, rarity (how many S2/S3 records share the
  record's rarest key), S1 duplicates (how many reference entities share the key), competition
  (best address fit of the record among OTHER S1 with its exact name, and our margin over it).
  No country one-hot (France is unseen in training).

**Model type:** LightGBM binary classifier (MIT licence), up to 3000 rounds with early stopping on a
disjoint S1 set; a second LightGBM stacker trained on the matcher's out-of-sample validation scores
(score shape within the S1 + a few raw pair features).  
**Threshold selection method:** one-to-one resolution, then threshold (+ relative-to-best) chosen
for macro F0.5 including singletons, cross-fitted (tune on half the validation S1, report on the other).

---

## 5. Results & Error Analysis

Validation = 60,000 train S1 never used for fitting, scored against all train S2/S3.

| version | macro F0.5 |
|---|---:|
| first pipeline (baseline) | 0.9305 |
| + blocking v2 (11 keys, learned ranker) | 0.9558 |
| + token-difference features | 0.9776 |
| + rarity, S1-duplicate features, longer training | 0.9807 |
| + competition features | 0.9814 (cross-fitted) |
| **+ stacker (final)** | **0.9816 (cross-fitted)** |

- **Precision / recall (final, validation):** ~0.994 / ~0.957.
- **Test run (submitted files):** within the submission time box the final pipeline (E9) processed
  255,000 of the 1,732,544 test S1 (14.7%: 119,226 India, 97,709 US, 38,065 France); the other
  1,477,544 S1 carry the first pipeline's validated rows, from which 2,063 matches to records already
  assigned by E9 were removed (one-to-one). 5,268,060 matches in total; 129,533 S1 predicted as
  singletons; the official validator passes. The validation scores above are NOT leaderboard scores:
  the expected test score is a mix of the E9 validation level (14.7% of S1) and the first pipeline's
  (0.9305 on validation, 85.3% of S1).
- **Common false positives:** a record owned by a same-name S1 at a neighbouring address; one name
  word substituted; random-brand record at a shared address.
- **Common false negatives:** records with an EMPTY address whose name is shared by several S1
  (median 36) - provably indistinguishable, 1.46% of true pairs; names replaced by pseudo-brands;
  house numbers changed by noise.
- **Measured ceiling:** indistinguishable records alone cap any compliant system at 0.9956; the
  matcher's probabilities are calibrated (p 0.3-0.65 -> 46% true regardless of entity context).

---

## 6. Conclusion
Most of the gap to a perfect score came from candidate generation (a count-based pre-cap) and from
features that could not tell deliberate edits from noise; fixing both lifted validation macro F0.5
from 0.9305 to 0.9816. The remaining error is dominated by records that the provided fields cannot
attribute (empty address + shared name, pseudo-brand at a shared address).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` - `README.md` has the exact commands. Entry points:
`src.preprocessing.preprocess` (normalise), `src.preprocessing.build_indexes --index-name index_v2`
(blocking indexes), `src.train --config configs/pipeline_v8.yaml` (ranker + matcher),
`src.fit_stacker` (stacker), `src.predict --config configs/pipeline_final.yaml` (test candidates +
scores), `src.final_submission` (decision, one-to-one, output files, validator).

### B. Additional Results
`docs/EXPERIMENTS_V2.md` (audit, error budgets, experiments E1-E9, ceiling analysis),
`experiments/results/<exp>/` (reports, tuning grids, feature importance).
