# Resume Screening Agent Notes

## Dataset layout
- `resumes/` — raw resume PDFs (kept out of git via `.gitignore`). Drop any new data here.
- `data/` — use for small, structured artifacts you want in git (e.g., `data/metadata.csv`, `data/processed/*.json`).
- `data/tmp/` or other scratch dirs — fine for local work; add them to `.gitignore` if you create them.

## Git hygiene
- `.gitignore` excludes `resumes/` and `*.docx` so large/raw resumes stay local.
- Keep only lightweight metadata, prompts, and configs under version control.
- If you add new large datasets or embeddings, extend `.gitignore` accordingly.

## Agent workflow (baseline)
1) Read PDFs from `resumes/`.
2) Extract text/metadata; write small structured outputs to `data/processed/`.
3) Train/evaluate or run screening using those processed artifacts; keep heavyweight caches out of git.
