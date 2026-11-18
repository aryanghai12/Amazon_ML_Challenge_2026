# Amazon Entity Resolution Challenge

A reproducible, modular entity-resolution pipeline for candidate generation, pairwise matching, evaluation, and submission artifact creation.

## Project priorities

1. High-recall and scalable candidate generation.
2. Efficient blocking rather than full all-pairs comparison.
3. Transparent and testable pairwise features.
4. Reproducible matching and aggregation.
5. Complete experiment and decision tracking.
6. Clear documentation for human contributors and AI coding agents.

## Repository workflow

- Do not commit directly to `main`.
- Create a feature branch for each task.
- Open a pull request with tests and documentation.
- Record meaningful experiments in `experiments/experiment_registry.csv`.
- Update `docs/07_AI_AGENT_CONTEXT.md` after every substantial agent session.
- Never commit credentials, private data, or unrestricted generated artifacts.

## Suggested setup

```bash
python -m venv .venv
# Activate the environment
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

Run the pipeline: see *Full pipeline* below.

## Full pipeline (all members)

Running it on another machine: see [`RUN_ME_FIRST.md`](RUN_ME_FIRST.md).

See [`docs/PIPELINE.md`](docs/PIPELINE.md). Short version:

```bash
pip install -r requirements.txt
python -m src.preprocessing.preprocess --data-dir <DATASET_DIR>   # once
python -m src.pipeline                                          # indexes -> train -> predict -> submission v1
python -m src.make_submission --variant 2                       # v2..v5 in seconds
```

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv` (validated), reports in
`experiments/results/<exp_id>/`.

## Preprocessing (Member 2)

Data loading, EDA, normalisation and blocking-key analysis: see
[`docs/PREPROCESSING.md`](docs/PREPROCESSING.md), [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md)
and [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

```bash
python -m src.preprocessing.preprocess --data-dir <DATASET_DIR>   # -> data/processed/*.parquet
```

## Team ownership

| Member | Ownership |
|---|---|
| Member 1 | Features and matching |
| Member 2 | EDA and preprocessing |
| Member 3 | Documentation and AI-agent context |
| Member 4 | Blocking, evaluation, and integration |

## Current status

The repository contains the initial structure, documentation, contracts, and implementation roadmap. Dataset-specific schemas and algorithms must be finalized after inspecting the official challenge files.

## Final pipeline (v2, submitted) - exact reproduction

Audit, experiments and measured ceiling: [`docs/EXPERIMENTS_V2.md`](docs/EXPERIMENTS_V2.md).
Validation macro F0.5 (60k held-out train S1, cross-fitted threshold): **0.9816** (first version 0.9305).

```bash
pip install -r requirements.txt
# 1. normalise the raw TSVs (once)                                              ~20 min
python -m src.preprocessing.preprocess --data-dir <DATASET_DIR>
# 2. blocking indexes v2 (11 keys, cap applied at load time)                    ~7 min per split
python -m src.preprocessing.build_indexes --splits train,test --index-name index_v2 --max-block 3000 \
  --keys name_skel_pair,name_first,name_prefix5,num_street,state_name_skel,addr_tokens,name_compact,name_prefix8,addr_pair,name_addr,num_street1
# 3. stage-2 ranker + matcher (E8 configuration)                                ~35 min
python -m src.train --config configs/pipeline_v8.yaml --exp-id E8_competition --use-cache
# 4. entity-aware stacker on the matcher's out-of-sample validation scores      ~5 min
python -m src.fit_stacker --config configs/pipeline_final.yaml --exp E8_competition
# 5. test candidates + scores (models/v8). Full run ~2 h on a 20-thread laptop;
#    --deadline-min N stops early and the remaining S1 fall back to the first pipeline's rows
python -m src.predict --config configs/pipeline_final.yaml --tag v2test [--deadline-min 65]
# 6. stacker + one-to-one decision, write output/*.tsv, run own checks + official validator
python -m src.final_submission --config configs/pipeline_final.yaml --tag v2test
```

The first-pipeline fallback rows come from `models/baseline_M20260927_1454/` (config
`models/baseline_M20260927_1454/pipeline.yaml`, `python -m src.predict` + `python -m src.make_submission --variant 1`),
kept as `data/interim/*_baseline_M20260927_1454.tsv`.

### Submitted files (2026-09-27)
`output/matching_results.tsv` / `output/candidate_pairs.tsv`: 1,732,544 S1 rows each; 255,000 S1 (14.7%)
from the final pipeline (E9, `models/v8` + stacker, `configs/pipeline_final.yaml`), 1,477,544 from the
first pipeline's validated rows (2,063 of its matches removed for one-to-one). 5,268,060 matches.
Official validator: PASS. Details: `data/interim/final_submission.json`, `experiments/logs/final_submission.log`.
Validation macro F0.5 (0.9816 E9 / 0.9305 first pipeline) is not a leaderboard score.
