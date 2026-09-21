# CLAUDE.md

Self-contained research repo. References only itself — no paths or data
dependencies on other repos.

## Hard rules
- **Never run git commit, git push, or any git write.** Francisco is the sole
  committer. Propose changes; he commits.
- **Never commit data, PDFs, keys, the venv, or ROADMAP.md** — all gitignored.
- Raw data enters only via `scripts/build_panel.py` → `data/cache/prices.parquet`.
- Don't change the locked harness in `src/maplab/contract.py` / `backtest.py`
  without being asked.

## Style
Step by step. Assess before changing. Minimal, scoped edits — no refactors or
new files unless asked. The plan lives in ROADMAP.md (local); read it if present.
