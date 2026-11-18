"""Build the final submission zip (structure required by the challenge README).

<name>.zip
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/{src, configs, scripts, tests, docs, models (final + fallback),
                                   experiments (reports), README.md, requirements.txt, ...}
  Documentation_template.md

    python -m scripts.package_submission --name <team_name>_submission.zip
"""
from __future__ import annotations

import argparse
import glob
import os
import zipfile

CODE = "code/business_entity_resolution"
KEEP_MODELS = ["models/v8", "models/baseline_M20260927_1454"]
RESULTS = ["M20260927_1454", "E1_blockv2", "E2_diff", "E3_rarity", "E4_s1dup", "E5_rounds", "E5_seed7",
           "E6_stacker", "E7_brand", "E8_competition", "E8_half", "E9_E8stacker", "E8_competition_stacker"]


def files():
    out = []
    for pat in ("src/**/*.py", "configs/*.yaml", "configs/*.json", "scripts/*.py", "scripts/*.sh", "scripts/*.bat",
                "tests/*.py", "docs/**/*.md", "docs/assets/**/*.png"):
        out += glob.glob(pat, recursive=True)
    for m in KEEP_MODELS:
        out += [p for p in glob.glob(f"{m}/*") if os.path.isfile(p)]
    for r in RESULTS:
        out += [p for p in glob.glob(f"experiments/results/{r}/*") if os.path.isfile(p)
                and not p.endswith(".parquet")]
    out += ["experiments/experiment_registry.csv", "README.md", "requirements.txt", "requirements.lock.txt",
            "pyproject.toml", ".gitignore", "data/interim/final_submission.json"]
    return sorted({p.replace("\\", "/") for p in out if os.path.exists(p) and "__pycache__" not in p})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="submission.zip")
    a = ap.parse_args()
    with zipfile.ZipFile(a.name, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in ("output/matching_results.tsv", "output/candidate_pairs.tsv"):
            z.write(p, p)
        z.write("Documentation_template.md", "Documentation_template.md")
        fl = files()
        for p in fl:
            z.write(p, f"{CODE}/{p}")
    print(f"wrote {a.name}: {len(fl) + 3} files, {os.path.getsize(a.name) / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
