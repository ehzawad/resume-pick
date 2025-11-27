# Repository Guidelines

## Project Structure & File Organization
- The repository is a flat collection of resumes at the project root; expect `.pdf`, `.docx`, and occasional `.png` assets. There are no subdirectories, build outputs, or source modules.
- Keep new additions at the root unless there is a clear, shared need for a subfolder (e.g., `archived/` for superseded files); propose any new structure in a PR description.
- Prefer clear, unique filenames so contributors can spot the owner and revision at a glance.

## Adding or Updating Resumes
- Normalize names to `Lastname_Firstname_Role_Version.ext` (e.g., `Rahman_Arif_FullStack_v2.pdf`); avoid spaces where possible to make shell usage easier.
- Prefer `.pdf` for the canonical copy; if a `.docx` is required, add the matching PDF alongside it.
- When replacing a file, keep the prior version until reviewers confirm the swap; mark the new one with a higher version tag.
- Do a quick sanity check before committing: open the document locally to ensure it is readable and not password-protected.

## Build, Test, and Development Commands
- There is no build pipeline. Useful local checks:
  - `find . -maxdepth 1 -type f | wc -l` — count tracked documents.
  - `file "*.pdf"` — confirm mime types after bulk renames.
  - `md5 <file>` — compare hashes when verifying duplicate submissions.

## Naming Conventions & Metadata
- Stick to ASCII characters in filenames; if a name requires non-ASCII characters, include an ASCII transliteration for searchability.
- Use underscores for separators and short role tags (`FullStack`, `Data`, `Mobile`) rather than long sentences.
- Avoid mixing date formats; if needed, use `YYYYMMDD` at the end (`..._20240915.pdf`) for chronological sorting.

## Testing & Validation
- Before submitting, open a sampling of modified files to confirm fonts render and pages are intact.
- If deduplicating, compare hashes or visually diff PDFs to avoid dropping distinct revisions.
- For bulk operations, run `rg --files` to ensure only intended filenames changed; avoid accidental deletions in this flat structure.

## Commit & Pull Request Guidelines
- Commit messages: concise and action-led. Examples: `chore: add Rahman full-stack resume`, `chore: replace Akter QA resume v2`.
- Pull requests should list added, updated, and removed files explicitly, and note whether any originals are archived.
- If a file is removed or redacted, state the reason (e.g., duplicate, corrupted, owner request) in the PR description.

## Security & Handling
- Files contain personal information; do not publish them externally or upload to third-party tools without consent.
- Avoid embedding credentials or signatures in filenames or commit messages; keep sensitive details inside the documents only.
