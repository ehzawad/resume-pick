#!/usr/bin/env bash
# Run the RSAS pipeline end-to-end with caching reuse on repeated job_id.

set -euo pipefail

# Configurable knobs (env overrides allowed)
JOB_ID="${JOB_ID:-live-full}"
DESC="${DESC:-sample_jobs/senior_ml_engineer.txt}"
RESUMES="${RESUMES:-resumesets}"
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

mkdir -p "$(dirname "$OUT_JSON")"

echo "Running RSAS pipeline..."
"$PYTHON_BIN" rsas_cli.py process \
  -j "$JOB_ID" \
  -d "$DESC" \
  -r "$RESUMES" \
  -o "$OUT_JSON"

echo "Exporting rankings to CSV..."
"$PYTHON_BIN" rsas_cli.py ranking \
  -j "$JOB_ID" \
  -n 20 \
  -e "$OUT_CSV"

echo "Done."
echo "JSON: $OUT_JSON"
echo "CSV:  $OUT_CSV"
