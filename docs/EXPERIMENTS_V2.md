# Pipeline v2: audit, bottlenecks, experiments

Branch `exp/ceiling-audit`. Baseline = experiment `M20260927_1454` (valid macro F0.5 **0.9305113266**).
Every number below is measured on the SAME validation set as the baseline: the 60,000 train S1
selected by `src.train` (seed 42), disjoint from the 150,000 fit S1 and 15,000 early-stop S1,
scored against ALL train S2/S3 records (real distractors). No test labels exist or are used.

## 1. Audit of the baseline

### Where the 0.0695 was lost (sequential counterfactual attribution, `scripts/audit_matcher.py`)

| error source | F0.5 lost | pairs |
|---|---:|---:|
| blocking: true pair shares no key | 0.0134 | 5,635 |
| blocking: dropped by the `pre_k=100` shared-key-count cap | 0.0236 | 11,210 |
| blocking: dropped by the `top_k=40` re-rank | 0.0017 | 926 |
| matcher: true pair in candidates, p below threshold | 0.0204 | 10,143 |
| matcher: removed by the relative rule | 0.0027 | 1,876 |
| false merges on S1 that have matches | 0.0077 | 2,243 |
| false merges on singletons | 0.0038 | 254 |

The **perfect-matcher ceiling of the baseline candidates was 0.9647**, so 0.99 was impossible
without changing candidate generation.

Root causes found in the code:
* `pre_k` ranks the stage-1 union by the NUMBER of shared keys; ties (most pairs share 1 key) are
  broken by encoded candidate id, i.e. S2 before S3 and low rows first -> S3 recall 0.897 vs S2 0.932.
* Blocks above a size cap are DROPPED: generic names ("Pediatric Associates", "Vision ...") and
  `addr_tokens` (cap 200: 24M records in oversized blocks) lose every key.
* `num_street` ignores 1-digit house numbers; no key covers "name replaced by a random brand,
  same address", domain / hashtag names (`visionphysicians.com`) or native-script names.
* Pair features compare numbers and tokens as SETS: a truncated house number (136 -> 13, true-match
  noise) and a changed one (3317 -> 3319, distractor) look the same, as do a misspelt word and a
  substituted word.
* Decision layer: threshold + relative + one-to-one is already near-optimal; an expected-F0.5
  per-entity rule does not beat it (0.93037 vs 0.93051, cross-fitted).

### Validation fidelity
* No leakage: fit / early-stop / validation S1 are disjoint; S2/S3 records are shared (as on test).
* Open world: only the 60k validation S1 compete in one-to-one; on test all S1 compete. 38% of the
  baseline false merges were records owned by an S1 outside the validation set.
* France (15% of test S1) is absent from training. Predicted match volume on test is the same for
  France as for US / India (3.43 / 3.30 / 3.23 matches per S1; train ground truth 3.46).

## 2. Changes

| # | change | files |
|---|---|---|
| E1 | **blocking v2**: 11 keys (+ `name_compact`, `name_prefix8`, `addr_pair`, `name_addr`, `num_street1`), caps 1000 applied at load time; NO shared-key-count pre-cap; learned **stage-2 ranker** (LightGBM on cheap signals, trained on fit S1) scores every stage-1 pair, top 40 kept | `preprocessing/keys.py`, `blocking/candidates.py` (`SimCandidateGenerator`), `blocking/ranker.py`, `scoring.py` |
| E2 | **token-difference features**: exact / fuzzy / substituted name and address tokens, number exact / truncated / 1-edit / unmatched / min numeric gap, frequency rank of substituted words, legal-form flags | `features/diff_features.py` |
| E3 | **rarity**: how many S2/S3 records share the record's rarest key | `features/rarity.py` |
| E4 | **S1 duplicates**: how many S1 share the key (for the candidate: how many reference entities compete for that record) | `features/rarity.py` |
| E5 | matcher up to 3000 rounds with early stopping (600 never early-stopped) | config |

A LightGBM pitfall found on the way: with ~0.1% positives (stage-2 ranker) plain `binary` training
diverges (train logloss rises; AUC 0.92 < a single raw feature's 0.94). `max_delta_step=1` and
`lambda_l2=10` fix it (AUC 0.9996).

## 3. Results (same validation set)

| experiment | macro F0.5 | delta | candidate recall | perfect-matcher ceiling | false merges | in-candidate misses |
|---|---:|---:|---:|---:|---:|---:|
| baseline M20260927_1454 | 0.9305 | - | 0.9143 | 0.9647 | 2,497 | 12,029 |
| E1 blocking v2 | 0.9558 | +0.0253 | 0.9908 | 0.9970 | 2,917 | 16,946 |
| E2 + diff features | 0.9776 | +0.0218 | 0.9908 | 0.9970 | 1,474 | 8,507 |
| E3 + rarity | 0.9792 | +0.0016 | 0.9908 | 0.9970 | 1,413 | 7,376 |
| E4 + S1 duplicates | 0.9804 | +0.0012 | 0.9908 | 0.9970 | 1,155 | 7,259 |
| **E5 + longer training** | **0.9807** | +0.0003 | 0.9908 | 0.9970 | 1,162 | 7,054 |

Threshold tuned on the same validation set: cross-fitted (tune on half A, report on half B and vice
versa) E5 scores 0.98059, i.e. tuning optimism is ~0.0001.

### Targeted residual experiments (after E5; same validation S1 and candidates)

| experiment | macro F0.5 (tuned) | cross-fitted | delta vs E5 (paired bootstrap, 95% CI) | false merges | in-cand. misses |
|---|---:|---:|---|---:|---:|
| E5 (reference) | 0.98073 | 0.98059 | - | 1,162 | 7,054 |
| E5, seed 7 (noise check) | 0.98080 | 0.98068 | +0.00007 [-0.00022, +0.00037] | - | - |
| E6 stacker on E5 (entity-aware 2nd stage) | 0.98108 | 0.98108 | ~+0.0005 | - | - |
| E7 + pseudo-brand likelihood | 0.98100 | 0.98064 | +0.00028 [-0.00008, +0.00061] n.s. | 1,156 | 6,880 |
| **E8 + competition (same-name S1)** | 0.98149 | 0.98138 | **+0.00077 [+0.00038, +0.00113]** | **899** | 7,306 |
| E8 on half the fit S1 (learning curve) | 0.98064 | 0.98054 | E8 - half: +0.00085 [+0.00052, +0.00118] | - | - |
| **E9 = E8 + stacker (FINAL)** | 0.98173 | **0.98173** (0.98162 with the test row filter) | **+0.00100 [+0.00054, +0.00143]** | - | - |
| E10 + name/legal + building competition | not finished | - | dropped: +~50 min test inference | - | - |

Entity-level decision audit (E5): an oracle choosing the best cut-off per S1 on E5's ranking reaches
0.9945, or 0.9922 when the indistinguishable pairs cannot be used - but E5's probabilities are
calibrated GIVEN the entity context (pairs with 0.3 <= p < 0.65 are 46% true whether or not a
score cliff lies below them), so this headroom is label knowledge, not a decision-rule defect; the
stacker's gain comes from its raw pair features, not from the score shape (0.98130 without them).

Per-category oracle gain (E5; fix only that category's errors): indistinguishable 0.0039 (not
recoverable), empty other address 0.0032, house number 0.0035, name replaced 0.0025, name word
substituted 0.0014, native script 0.0012, other 0.0008.

### Test-time protocol (submission)
Full v2 test inference takes ~2 h (stage-1 safety cap 1200) to ~4 h (4000) on the team laptop.
Within the submission time box the final pipeline (E9) scored the first test S1 (see
`data/interim/final_submission.json`); the remaining S1 keep the validated baseline rows (records
already matched by an E9 S1 are removed from them - one-to-one across both parts).

## 4. Why not 0.99: the measured ceiling

1. **Indistinguishable records** (`scripts/ambiguity_bound.py`). 50.7% of S1 records share their
   exact name with another S1 of the same country. When an S2/S3 record has an EMPTY address and its
   owner's raw name is shared by other S1 (median 36 of them), no function of the provided fields can
   tell which S1 owns it. 1.46% of validation true pairs are of this kind; an oracle that knows every
   other true pair reaches at most **0.9956** (leaving them out beats guessing: 0.9876).
2. **Blocking**: 0.9% of true pairs are not candidates (ceiling 0.9970 with a perfect matcher).
3. E5 remaining loss 0.0193 = blocking 0.0030 + indistinguishable 0.0040 + learnable matcher errors
   0.0123. A matcher that fixed every learnable error would score 0.9930.
4. Remaining learnable errors are dominated by (a) empty-address records whose name is only nearly
   unique, (b) names replaced by a random brand where only the address links the records - a
   distractor at the same address looks the same, and (c) house-number edits that are sometimes
   noise and sometimes a distractor.

## 5. Runtime / memory (laptop: Ryzen, 24 GB RAM, RTX 4060)

| stage | baseline | v2 |
|---|---:|---:|
| train split indexes (once) | ~2 min | 6.6 min (index_v2, 11 keys) |
| training run (fresh candidates) | 15 min | 35 min; 12 min with cached candidates |
| training peak memory | 9.4 GB | 11-14.6 GB |
| test inference | 43 min | ~3.9 h (candidates saved; rescoring ~45 min) |

## 6. Reproduce

See the final section of the session report / `README.md` ("Pipeline v2").
