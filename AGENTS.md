# RSAS Agent Handbook

Keep this repo lean, reproducible, and safe to run. Treat resumes as local-only inputs and keep structured artifacts small.

## Repo map
- `rsas/` — agents, pipeline, storage (JSON object store), search, KB ingestion.
- `config/` — defaults and overrides (`RSAS_*` env vars can override).
- `resumesets/` — local PDFs; ignored by git.
- `data/processed/` — object store output (job profiles, traces, rankings).
- `sample_jobs/` — example job descriptions.
- `tests/` — unit/e2e stubs; `RSAS_TEST_MODE=1` uses mocked agents.

## Workflow
1) Install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
2) Offline runs: `RSAS_TEST_MODE=1` to bypass OpenAI.
3) Process resumes: `python rsas_cli.py process -j JOB_ID -d sample_jobs/role.txt -r resumesets/ -o results.json`.
4) Check status/rankings: `python rsas_cli.py status -j JOB_ID` / `python rsas_cli.py ranking -j JOB_ID -n 10`.
5) Search KB: `python rsas_cli.py search "query" --job-id JOB_ID` (Chroma optional; falls back to keyword scoring).

## Data hygiene
- Do not commit raw resumes, DOCX, or large embeddings. `.gitignore` already excludes `resumes/` and `*.docx`; extend if needed.
- Persist only small JSON/CSV outputs in `data/` (or `data/tmp/` for scratch and ignore it).
- Never commit secrets; set `OPENAI_API_KEY` in env when using real API.

## Testing & quality
- Smoke tests: `pytest -v` (uses mocks in `tests/e2e` and `tests/unit`).
- Type/style (optional): `mypy rsas/`, `ruff check rsas/`, `black rsas/`.
- For KB ingestion dry-run: `RSAS_TEST_MODE=1 python test_kb_ingestion.py`.

## Safety notes
- Pipeline is idempotent via hashed inputs and JSON traces. Stored under `data/processed/{job_id}`.
- Keep concurrency reasonable (`pipeline.max_concurrent_resumes` in config); default is 10.
- If you add new caches or large artifacts, update `.gitignore` and note them here.
