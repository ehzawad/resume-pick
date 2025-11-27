#!/usr/bin/env bash
# Run the RSAS pipeline on a subset of resumes to limit cost.

set -euo pipefail

JOB_ID="${JOB_ID:-live-50}"
DESC="${DESC:-sample_jobs/senior_ml_engineer.txt}"
SOURCE_RESUMES="${RESUMES:-resumesets}"
LIMIT="${LIMIT:-50}"
SUBSET_DIR="${SUBSET_DIR:-/tmp/rsas-subset-$JOB_ID}"
OUT_JSON="${OUT_JSON:-results/${JOB_ID}.json}"
OUT_CSV="${OUT_CSV:-results/${JOB_ID}.csv}"
PYTHON_BIN="${PYTHON_BIN:-/Users/ehz/venv-multi-agent/bin/python}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it and rerun." >&2
  exit 1
fi

if [[ -n "${RSAS_TEST_MODE:-}" ]]; then
  echo "Warning: RSAS_TEST_MODE is set; unset it for live LLM calls." >&2
fi

echo "Preparing subset of $LIMIT resumes from $SOURCE_RESUMES -> $SUBSET_DIR"
export SOURCE_RESUMES SUBSET_DIR LIMIT
"$PYTHON_BIN" - <<'PY'
import os, shutil, pathlib
from itertools import islice
source = pathlib.Path(os.environ["SOURCE_RESUMES"])
subset = pathlib.Path(os.environ["SUBSET_DIR"])
limit = int(os.environ["LIMIT"])
subset.mkdir(parents=True, exist_ok=True)
count = 0
for path in islice(sorted(p for p in source.iterdir() if p.is_file()), limit):
    shutil.copy(path, subset / path.name)
    count += 1
print(f"Copied {count} resumes to {subset}")
PY

mkdir -p "$(dirname "$OUT_JSON")"

echo "Running RSAS pipeline on subset..."
"$PYTHON_BIN" rsas_cli.py process \
  -j "$JOB_ID" \
  -d "$DESC" \
  -r "$SUBSET_DIR" \
  -o "$OUT_JSON"

echo "Exporting rankings to CSV..."
"$PYTHON_BIN" rsas_cli.py ranking \
  -j "$JOB_ID" \
  -n 20 \
  -e "$OUT_CSV"

echo "Done."
echo "Subset dir: $SUBSET_DIR"
echo "JSON: $OUT_JSON"
echo "CSV:  $OUT_CSV"
