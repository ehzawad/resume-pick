# Resume Sorting Agent System (RSAS)

AI-driven resume ranking pipeline with multi-agent orchestration, structured outputs, and file-based persistence.

## What it does
- Parses PDFs, extracts skills, matches to a job profile, scores, ranks, and runs a basic bias check.
- Uses OpenAI `gpt-5.1` (default) for structured responses; override via config/env.
- Idempotent by design: agent outputs are cached by job + content hashes to avoid repeat LLM calls.
- Stores artifacts in the object store under `data/processed/<job_id>/`; vectors persist via Chroma (if enabled).

## Requirements
- Python 3.11+
- OpenAI API key (`OPENAI_API_KEY`)
- pdfplumber + PyPDF2 (already in requirements)

## Quick start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=...       # required
unset RSAS_TEST_MODE            # ensure live calls

# Process the current 50 PDFs in resumesets/
/Users/ehz/venv-multi-agent/bin/python rsas_cli.py process \
  -j live-50 \
  -d sample_jobs/senior_ml_engineer.txt \
  -r resumesets \
  --limit-resumes 50 \
  -o results/live-50.json

# View rankings / export CSV
/Users/ehz/venv-multi-agent/bin/python rsas_cli.py ranking \
  -j live-50 -n 20 -e results/live-50.csv
```

## Configuration
- Base config: `config/default.yaml`
  - `openai.model`: `gpt-5.1` (change via `RSAS_OPENAI_MODEL`)
  - `storage.object_store_dir`: `data/processed`
  - `pipeline.max_concurrent_resumes`: 10
  - `pipeline.idempotency`: true (traces cached per job + content hash)
- Env overrides: any `RSAS_*` path, e.g. `RSAS_OPENAI_MODEL`, `RSAS_PIPELINE_MAX_CONCURRENT_RESUMES`.

## Security

### API Key Protection
**CRITICAL:** Never commit your OpenAI API key to the repository!

**Protected files (in `.gitignore`):**
- `.env` - Environment variables (contains `OPENAI_API_KEY`)
- `.env.local`, `.env.*.local` - Local environment overrides
- `config/local.yaml`, `config/secrets.yaml` - Local config overrides
- `*.key`, `*.pem`, credentials files

**Best Practices:**
1. Copy `.env.example` to `.env` and add your API key:
   ```bash
   cp .env.example .env
   # Edit .env and set: OPENAI_API_KEY=sk-...
   ```
2. Always use environment variables (`OPENAI_API_KEY`), never hardcode keys
3. Before committing, verify no keys exposed:
   ```bash
   grep -r "sk-" . --exclude-dir=.git
   git diff --staged | grep "sk-"
   ```
4. The config uses `api_key_env: "OPENAI_API_KEY"` to reference the env var safely

**If you accidentally commit a key:**
1. Revoke it immediately at https://platform.openai.com/api-keys
2. Generate a new key
3. Use `git filter-branch` or BFG Repo-Cleaner to remove from history

## Data & caching
- Resumes: place PDFs in `resumesets/` (currently 47 valid files after removing 3 failed ones).
- Object store layout: `data/processed/<job_id>/...` (parsed resumes, profiles, scorecards, rankings, traces, KB).
- Content-hash-aware parsing: unchanged PDFs reuse cached outputs even if renamed.
- To force a fresh run: use a new `job_id` or delete `data/processed/<job_id>/`.

## Testing
```bash
/Users/ehz/venv-multi-agent/bin/python -m pytest -q
```

## Notes
- The subset runner `scripts/run_subset_pipeline.sh` defaults to `JOB_ID=live-50`, `LIMIT=50`; override envs to change the slice.
- Structured-output prompts enforce JSON-only responses; malformed outputs fall back to deterministic ranking instead of failing the pipeline.***
