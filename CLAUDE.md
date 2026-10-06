# CLAUDE.md

Self-contained research repo. References only itself — no paths or data
dependencies on other repos.

## Hard rules
- **Never run git commit, git push, or any git write.** Francisco is the sole
  committer. Propose changes; he commits.
- **Never commit data (except data/cache/prices.parquet), PDFs, keys, the venv, or ROADMAP.md** — all gitignored.
- Raw data enters only via `scripts/build_panel.py` → `data/cache/prices.parquet`.
- Don't change the locked harness in `src/maplab/contract.py` / `backtest.py`
  without being asked.

## Style
Step by step. Assess before changing. Minimal, scoped edits — no refactors or
new files unless asked. The plan lives in ROADMAP.md (local); read it if present.

## Model & notebook conventions
- New models subclass `maplab.models.Strategy`; `family` must be a
  `plotting.FAMILY_COLORS` key; LONG_SHORT weights pass `_validate` unclipped.
- SLSQP everywhere: `ftol=1e-12`, `maxiter=1000`, analytic jac for objective
  and constraints; no silent fallbacks — log and record
  (`fallback_dates` / `retry_dates`), raise on real failure.
- Estimation on LOG returns via `Panel.slice`; `mu = mean*252 + 0.5*diag(Sigma)`.
- Use fresh strategy instances per backtest.
- Sharpe and tangency use BIL daily returns as a time-varying risk-free rate
  (load_rf_returns); rf is a required argument.
- Notebooks: execute with `nbconvert --execute --inplace`; HTML exports go to
  `~/Desktop`, never into the repo.
- Notebooks are committed with outputs cleared (run, verify, export HTML
  outside the repo, then `nbconvert --clear-output`).
