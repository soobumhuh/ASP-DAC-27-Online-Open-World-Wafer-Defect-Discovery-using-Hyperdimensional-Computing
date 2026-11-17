#!/bin/bash
# Post-processing of one experiment arm: nested CV (run_cv_clean) -> native-grid evaluation -> paired bootstrap vs the
# reference (best_clean_adc, native cases) -> one summary line in results/sota/summary.tsv.
#   scripts/sota_post.sh <config> <tag> [run_cv_clean args: --extraction f.json --set k=v ...]
set -e
cd "$(dirname "$0")/.."
CFG=$1; TAG=$2; shift 2
OUT=$(.venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['paths']['out'])")
REF=${SOTA_REF:-results/frontend_a3s1_eval/native_cases_clean_adcrim.pkl}
mkdir -p $OUT
.venv/bin/python scripts/run_cv_clean.py --config $CFG --tag "$TAG" "$@" > $OUT/cv_clean$TAG.log 2>&1
.venv/bin/python scripts/native_eval.py --config $CFG --cases $OUT/cv_clean${TAG}_cases.pkl --stage merged --workers 12 \
  --out $OUT/native$TAG.json --save-cases $OUT/native_cases$TAG.pkl > $OUT/native$TAG.log 2>&1
.venv/bin/python scripts/bootstrap_ci.py --runs ref=$REF:merged arm=$OUT/native_cases$TAG.pkl:merged --ref ref --n 1000 \
  --out $OUT/boot$TAG.json > $OUT/boot$TAG.log 2>&1
.venv/bin/python scripts/sota_summary.py --arm "$(basename $OUT)$TAG" --native $OUT/native$TAG.json --boot $OUT/boot$TAG.json \
  --cv $OUT/cv_clean$TAG.json --args "$*" >> results/sota/summary.tsv
tail -1 results/sota/summary.tsv
