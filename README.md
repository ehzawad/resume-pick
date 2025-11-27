# Resume Sorting Agent System (RSAS)

AI-driven resume ranking pipeline with multi-agent orchestration, structured outputs, and file-based persistence.

## What it does
- Parses PDFs, extracts skills, matches to a job profile, scores, ranks, and runs a basic bias check.
- Uses OpenAI `gpt-5.1` (default) for structured responses; override via config/env.
- Idempotent by design: agent outputs are cached by job + content hashes to avoid repeat LLM calls.
- Stores artifacts in the object store under `data/processed/<job_id>/`; vectors persist via Chroma (if enabled).

## How It Works

The system operates in two phases:

**Phase 1: Main Pipeline (`process` command)**
1. Input: 1 job description file + N resume PDFs
2. Process: Job description → JobProfile, each resume → parsed → skills extracted → matched/scored against job
3. Output: Ranked candidates saved to `data/processed/<job_id>/`

**Phase 2: Knowledge Base Ingestion (Auto-runs if `kb.auto_ingest: true`)**
1. Creates summaries of each candidate (using GPT-5.1)
2. Generates embeddings (text-embedding-3-small, hardcoded in openai_client.py)
3. Stores in `data/processed/<job_id>/kb/` + ChromaDB (if enabled)
4. Enables semantic search functionality

**Key Distinction:**
- `process` = Match candidates to a **specific job description** → ranked results for that job
- `search` = Find candidates by **any criteria** (independent of original job description)
- Both require `process` to run first, but serve different purposes

## Requirements
- Python 3.13+
- OpenAI API key (`OPENAI_API_KEY`)
- pdfplumber + PyPDF2 (already in requirements)

## Quick start
```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API key
export OPENAI_API_KEY=sk-your-key-here  # required
unset RSAS_TEST_MODE                     # ensure live calls

# 3. Process resumes against a job description
# - Reads job description from sample_jobs/senior_ml_engineer.txt
# - Processes up to 50 PDFs from resumesets/ folder
# - Scores and ranks ALL candidates, saves to results/live-50.json
python rsas_cli.py process \
  -j live-50 \
  -d sample_jobs/senior_ml_engineer.txt \
  -r resumesets \
  --limit-resumes 50 \
  -o results/live-50.json

# 4. View top 20 ranked candidates (all 50 are ranked internally)
# - Export full CSV with all 50 ranked candidates
python rsas_cli.py ranking \
  -j live-50 \
  -n 20 \
  -e results/all-50-ranked.csv

# 5. Semantic search through processed candidates (independent of job description)
# - Search by ANY criteria, not just what was in the original job posting
python rsas_cli.py search "5 years React experience" \
  -j live-50 \
  -n 10
```

## Understanding the Commands

### `process` - Run the Full Pipeline
Processes resumes for a job opening and ranks candidates.

**What it does:**
1. Reads job description file
2. Parses resume PDFs (up to `--limit-resumes` count, or all if not specified)
3. Scores and ranks **ALL** candidates against the job
4. Auto-generates KB (summaries + embeddings) for search
5. Saves results to `data/processed/<job_id>/`

**Key flags:**
- `--limit-resumes N`: Process at most N resumes (for cost control). All N are ranked.
- `-o FILE`: Save output JSON to FILE (parent dirs created automatically)

### `ranking` - View Results
Displays top N candidates from a completed job.

**What it does:**
- Shows top N candidates in terminal (for quick viewing)
- Exports **ALL** candidates to CSV (not just top N)

**Key flags:**
- `-n 20`: Show top 20 in terminal (default: 10)
- `-e FILE.csv`: Export full rankings to CSV (includes ALL processed candidates)

**Important:** `-n` is just for display. The CSV export contains all ranked candidates.

### `search` - Semantic Search
Search through processed candidates by any criteria.

**What it does:**
- **Requires `process` to have run first** (needs KB records)
- Searches by ANY criteria (independent of original job description)
- Uses embeddings for semantic matching (or keyword fallback)

**Example:**
```bash
# Search for "React developer" even if job was "ML Engineer"
python rsas_cli.py search "React developer" -j live-50 -n 10
```

**Key distinction:** Query is what you're looking for, NOT from job description file.

### `status` - Check Pipeline Progress
Shows current status of a running or completed pipeline.

### `chat` - Interactive Q&A
Ask questions about candidates using an interactive chat interface.

## Output Files

After running `process`, outputs are stored in `data/processed/<job_id>/`:

- `parsed_resumes/` - Extracted text from PDF files
- `profiles/` - Structured candidate data (CandidateProfile objects)
- `scorecards/` - Scores for each candidate against the job
- `rankings.json` - Final ranked list (all candidates)
- `kb/` - Knowledge base records (summaries + embeddings metadata)
- `traces/` - Agent execution logs (for debugging and idempotency)

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
python -m pytest -q
```

## Notes
- The subset runner `scripts/run_subset_pipeline.sh` defaults to `JOB_ID=live-50`, `LIMIT=50`; override envs to change the slice.
- Structured-output prompts enforce JSON-only responses; malformed outputs fall back to deterministic ranking instead of failing the pipeline.***
