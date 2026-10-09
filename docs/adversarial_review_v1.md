# Adversarial review of `maplab-phase3` (`polish`)

Date: 2026-10-06  
Audited commit: `489b807d425937aac582ae6b7b2bdf26cccc5821`  
Method: read-only review of the source repository; all execution and mutation work was done in fresh scratch clones.

## Executive summary

The numerical backtests are reproducible from the shipped parquet and the central walk-forward implementation does prevent same-date data from entering portfolio estimates. Those strengths do not rescue the research claims. The repository calls notebooks 04–07 “pre-registered” even though their registration files were created after all four notebooks, including a commit message already announcing verdicts. The nominal 2023+ “held-out” period is also explicitly admitted not to be sealed: the universe and Black–Litterman parameter were chosen with the full sample available. Those are not cosmetic provenance problems; they invalidate the advertised confirmatory interpretation.

I found 2 BLOCKER, 7 MAJOR, 8 MINOR, and 2 NIT findings. The most consequential non-provenance problems are incomplete accounting for research multiplicity, public Colab links that run an older branch without the claimed setup cell, questionable redistribution of vendor data, and thin tests around notebook 11 and turnover mechanics.

| Severity | Count |
|---|---:|
| BLOCKER | 2 |
| MAJOR | 7 |
| MINOR | 8 |
| NIT | 2 |
| **Total** | **19** |

## Findings

### F01 — BLOCKER — notebooks 04–07 were not pre-registered

- **Location:** `README.md:6`, `README.md:89-91`; `registrations/nb04.toml:1-6`, `nb05.toml:1-4`, `nb06.toml:1-4`, `nb07.toml:1-10`; notebook 04–07 hypothesis cells.
- **What is wrong:** The README badge and design section say hypotheses were fixed before tested statistics were computed. Git history proves the opposite for notebooks 04–07. All four TOMLs first appeared together at `00caa9d` on 2026-09-22 19:17:59, after the notebook commits: nb04 `66e9b29` 10:02:42, nb05 `21930c6` 10:52:57, nb06 `897c948` 13:04:03, and nb07 `210adef` 16:19:33. Each TOML even says it was “transcribed from notebook ... at commit 210adef.” The nb07 commit subject already records outcome language: “H1a not robust, H7 forecast bias supported.”
- **Evidence:** `git log --follow --diff-filter=A --format='%h %ad %s' --date=iso-strict -- <file>` produced the timestamps above. The TOMLs were then edited after results existed (`f6fe6d9`, `23f7d26`, `2be48d6`, `effc683`), including adding rule variants and setting `evaluated=true`.
- **Smallest fix:** Stop calling notebooks 04–07 pre-registered. Label them exploratory/post hoc, preserve the history disclosure prominently, and reserve “pre-registered” for 08–10 (and “descriptive” for 11). A new genuinely confirmatory study would require a new, future dataset and a registry frozen before any results.

### F02 — BLOCKER — the “held-out” 2023+ test window is contaminated by research decisions

- **Location:** `src/maplab/contract.py:41`; `README.md:17-19`, `README.md:86-93`, `README.md:114`; notebook 00 cell `5c005389`.
- **What is wrong:** Code and prose call 2023+ held out or a test window, but notebook 00 admits it “was not a sealed holdout” and that the universe and Black–Litterman `k` were set while the full sample was available. The universe is also a hand-picked set of surviving ETFs with uninterrupted history through 2026. The 2023+ results therefore cannot validate choices made after seeing those years.
- **Evidence:** Notebook 00 cell `5c005389` says exactly: “the universe and Black–Litterman's k were set while the full sample was available.” The same cell says `k` was chosen after a 2022-12-31 snapshot. The backtest itself is temporally clean, but research selection is not.
- **Smallest fix:** Rename 2023+ to a temporal subsample/consistency window everywhere and remove “held-out.” State which design choices saw which dates. For validation, freeze all choices and evaluate on subsequently arriving data or an untouched external universe.

### F03 — MAJOR — “No method beats equal weight” is not what the 30 tests establish

- **Location:** `README.md:17-19`, `README.md:25-29`; notebook 08 cell `68117c72`.
- **What is wrong:** Seven non-rejections concern Sharpe differences versus EW; the other 23 concern CAPM alpha versus zero. CAPM-alpha tests do not test whether a method beats EW. “30 of 30 null hypotheses hold” also turns failure to reject into affirmative support despite the repository's own 34–220-year power calculations.
- **Evidence:** `registrations/nb08.toml:19-23` defines the two distinct families. Notebook 11 reports that a true 0.2 Sharpe gap has only 12–51% power in 17 years. The careful phrase “not statistically distinguishable” at `README.md:17-19` is supportable; the bullet heading “No method beats” is not.
- **Smallest fix:** Replace the heading with “No registered Sharpe difference versus EW is detected.” Report the 7 Sharpe and 23 alpha families separately; do not sum them into a single evidentiary count.

### F04 — MAJOR — multiplicity correction covers only a selected subset of the research search

- **Location:** `registrations/nb08.toml:8`, `registrations/nb08.toml:27-32`; `README.md:30-31`; notebook 11 cell `n11-d4-read`.
- **What is wrong:** Holm/Romano–Wolf/DSR operate on the 23 retained Phase-1 runs. Designs rejected before backtest, universe construction, covariance/lookback choices, BL `k`, overlay choices, and multiple notebooks' snapshot-driven hypotheses are outside that family. The registration itself concedes the adjustment is a lower bound, but the README says the best run “survives deflation for the search” without that qualification.
- **Evidence:** `registrations/nb08.toml:31` explicitly says rejected designs are excluded and adjustment is a lower bound. Notebook 00 admits full-sample selection. The 23-run DSR is numerically 1.0000, but it answers only the artificially bounded family question.
- **Smallest fix:** Change “the search” to “the retained 23-run Phase-1 family.” Inventory all tried configurations or treat the total search count as unknown and avoid a definitive DSR survival claim.

### F05 — MAJOR — the public Colab badges do not run the audited notebooks

- **Location:** `README.md:68-82`; notebook setup cell `colab-setup` on `polish`.
- **What is wrong:** Every badge targets GitHub `main`, while this review and the claimed one-cell Colab setup are on `polish`. Public `main` is `bce2a7f`; `polish` is `489b807`. The public-main notebook 00 has no `colab-setup` cell and instead imports via a local parent-directory path. Thus “Click a badge ... first cell clones the repo ... no setup needed” is false for the linked branch.
- **Evidence:** `git ls-remote --heads ... main polish` returned distinct hashes. A scratch clone of public `main` showed notebook 00 cells `457fef1c`, `n00-setup`, then local imports—no clone/install cell. The `polish` notebook has `colab-setup`.
- **Smallest fix:** Merge `polish` to `main` before advertising the badges, or point every badge and the setup cell's clone to the exact reviewed branch/commit.

### F06 — MAJOR — the shipped EODHD panel appears incompatible with the cited personal-use terms

- **Location:** `README.md:57-62`, `README.md:161-164`; `data/cache/prices.parquet`; `LICENSE:5-13`.
- **What is wrong:** The repository publicly redistributes 4,611 × 17 adjusted-price observations. Merely saying the data are excluded from MIT does not grant redistribution rights. EODHD's current personal-use terms prohibit “retransmitting, redistributing, displaying, or granting access” to the information, original or repackaged. No commercial/redistribution licence is documented.
- **Evidence:** Official terms: <https://eodhd.com/financial-apis/terms-conditions>, “Personal and Commercial Use of Information,” lines 158–168 in the retrieved page. The parquet SHA and shape are pinned in `tests/test_data_file.py`.
- **Smallest fix:** Remove the panel unless the author can document a licence permitting redistribution. Otherwise provide a credentialed downloader/build recipe and distribute only derived, non-redistributable-safe artifacts approved by the vendor. This is a licence-compliance issue, not legal advice.

### F07 — MAJOR — the raw-data build is not reproducible on a fresh machine

- **Location:** `CLAUDE.md:3-4`, `CLAUDE.md:10`; `README.md:60-62`; `scripts/reshape_eodhd_archive.py:1-8`, `:19-20`; `scripts/build_panel.py:24-41`.
- **What is wrong:** The repo calls itself self-contained and says the panel was built by the scripts, but the first script hard-codes `~/Projects/research-data-eodhd` and a private manifest. A fresh user cannot reconstruct or audit the parquet from source. The hash test proves byte identity only, not provenance or correctness.
- **Evidence:** `scripts/reshape_eodhd_archive.py:19-20` requires the external archive. The fresh clone contains no `data/raw/eod_prices.csv`.
- **Smallest fix:** Document the external licensed input, its schema and immutable checksums; make the source path configurable; provide a reproducible acquisition path for licensed users. Remove “references only itself — no paths or data dependencies on other repos.”

### F08 — MAJOR — notebook-11 results are effectively untested

- **Location:** `tests/test_nb11_checks.py:1-33`; `notebooks/helpers/nb11.py:44-187`; `src/maplab/sharpe.py`.
- **What is wrong:** The only notebook-11 integration test checks sample lengths and 72 rounded Sharpe values. It never tests moments, PSR, MinTRL, power, clustering, effective rank, or DSR on the actual dataset.
- **Evidence:** In a disposable clone I replaced `dsr`'s probability with the constant `0.5`. `tests/test_sharpe.py::test_dsr_paper_example` failed, but `tests/test_nb11_checks.py::test_build_data_matches_registered_sharpe` still passed. A wiring error between the unit-tested formula and the notebook would therefore escape.
- **Smallest fix:** Add real-data assertions for every notebook-11 table, including the reported best run, `K`, `K_eff`, trial variance, expected maximum, DSR, PSR minima, MinTRL bounds, and power ranges.

### F09 — MAJOR — no test validates the backtest's turnover magnitude

- **Location:** `src/maplab/backtest.py:98-100`; `tests/test_turnover_window.py:7-33`; `tests/test_models.py` fixed-weight backtest test.
- **What is wrong:** Turnover tests validate window annualisation using a fabricated turnover series, but they do not independently calculate turnover from target and drifted weights. A factor-of-two error in the backtest's turnover formula survives the targeted tests.
- **Evidence:** Mutation `0.5 * |target-actual|` → `1.0 * |target-actual|` left `test_fixed_weight_backtest_trades_drift_back` plus both `test_turnover_window.py` tests passing (`3 passed`). Rounded reproduction gates would catch sufficiently large downstream cost changes, but the mechanic itself lacks an oracle.
- **Smallest fix:** Add a two-asset hand calculation with known pre-trade drift, target, one-way turnover, booked cost, and net return, including the initial cash-to-portfolio build.

### F10 — MINOR — initial deployment is charged as half a portfolio purchase

- **Location:** `src/maplab/backtest.py:92-100`, `:113`.
- **What is wrong:** `actual_w` starts at all zeros, then turnover is `0.5 * sum(abs(target-0)) = 0.5` for a fully invested long-only target. The half-L1 convention assumes both old and new books sum to one; from cash, purchases total 1.0. The first trade is charged 5 bp rather than the stated 10 bp per unit.
- **Evidence:** Direct algebra from the implementation. Annual turnover excludes the initial build, but `cost_daily` charges it, so this is a one-time 5 bp undercharge in every run.
- **Smallest fix:** Treat initial deployment separately as `sum(abs(target))`, or include cash explicitly in both old and new weight vectors.

### F11 — MINOR — the split assigns a rebalance's turnover and its cost to different windows

- **Location:** `src/maplab/backtest.py:102-114`; `notebooks/helpers/nb08.py:93-104`; `src/maplab/metrics.py:98-106`.
- **What is wrong:** The 2022-12-31 rebalance label belongs to train, but it is a Saturday, so its cost is booked on the first segment trading day, 2023-01-03, in test. Test-window returns include the cost while test-window turnover excludes the trade; train turnover includes it while train returns do not include its cost. Window cost-drag columns therefore do not reconcile to window net returns.
- **Evidence:** The 2022-12-31 turnovers range from 0.006107 (EW) to 0.118240 (MaxSharpe), so the shifted test cost ranges from 0.061 to 1.182 bp. The summary filters turnover on label dates, not charge dates.
- **Smallest fix:** Store turnover on its actual execution date or expose a cost series and compute window turnover/cost drag using execution dates.

### F12 — MINOR — maximum drawdown omits initial wealth

- **Location:** `src/maplab/metrics.py:47-50`; `notebooks/helpers/nb08.py:132-145`.
- **What is wrong:** Wealth begins at the first post-return value instead of 1.0. If a series starts with losses, drawdown from initial capital is ignored until a later peak.
- **Evidence:** `wealth = (1+r).cumprod()` followed by `wealth.cummax()` makes the first observation a zero drawdown by definition. Independent recomputation with a prepended 1.0 found no difference for the current reported full/train/test maxima, so this is latent rather than a current headline error.
- **Smallest fix:** Prepend initial wealth 1.0 (and do the same in drawdown-date/plot helpers); add a test whose first return is negative.

### F13 — MINOR — “Calmar” uses arithmetic mean return, not CAGR

- **Location:** `src/maplab/metrics.py:14-15`, `:53-55`; notebook 08 cell describing the performance table.
- **What is wrong:** The numerator is daily arithmetic mean × 252. Standard Calmar reporting usually uses annualized compounded return/CAGR. The notebook states “annualized return,” which is ambiguous, and readers will assume the standard definition.
- **Evidence:** Independent values show the difference is currently small but nonzero: EW arithmetic-Calmar 0.3941 vs CAGR-Calmar 0.3895; MaxSharpe 0.5354 vs 0.5315.
- **Smallest fix:** Rename it “arithmetic-return/MDD” or use CAGR and state the exact convention.

### F14 — MINOR — BIL is presented as risk-free cash although it is a traded ETF proxy

- **Location:** `README.md:45-46`, `README.md:99-100`; `src/maplab/data.py:45-64`; overlay documentation.
- **What is wrong:** BIL adjusted-price returns include ETF fees, price staleness/rounding, and short-duration mark-to-market risk. It is neither a risk-free rate nor frictionless cash/borrowing. The leveraged overlay sensitivity even finances at BIL with no borrowing spread.
- **Evidence:** The shipped BIL series has 1,779 zero-return days out of 4,610 returns, far more than other assets. All Sharpe, PSR and overlay cash legs inherit this proxy behavior.
- **Smallest fix:** Call BIL a cash proxy throughout, disclose fee/duration/staleness, and sensitivity-test a Treasury-bill yield/total-return series plus a borrowing spread for leverage.

### F15 — MINOR — dependence choices are too short for the adaptive strategy horizon and are not sensitivity-tested

- **Location:** `registrations/nb08.toml:14-16`, `registrations/nb10.toml:14`; `src/maplab/robust.py:19-37`, `:99-116`.
- **What is wrong:** Newey–West lags are 9/8/6 days and the stationary bootstrap mean block is 21 days, while weights are driven by overlapping 252-day estimates and held/rebalanced monthly. Those choices may miss slower dependence in strategy-return differences. The bootstrap resamples already-realized strategy paths rather than rerunning the adaptive estimator, so it conditions on the selected weight path.
- **Evidence:** The choices are fixed and implemented as registered, but no lag/block-length sensitivity is reported. This is most relevant to precise p-values and power/track-record estimates, not to the raw backtest means.
- **Smallest fix:** Report sensitivity to materially longer HAC lags and bootstrap blocks (for example 21/63/252), and distinguish conditional inference on realized strategy paths from a bootstrap that reruns allocation.

### F16 — MINOR — the effective-trial DSR variants do not implement the cited clustering route

- **Location:** `notebooks/helpers/nb11.py:164-187`; `src/maplab/sharpe.py:131-146`; notebook 11 cell `n11-d4-read`.
- **What is wrong:** The raw `K=23, V_all` DSR matches equations 28–30 and the paper's worked example. However, the clustering variant only replaces `K` with `nK=3` while reusing the variance across all 23 raw Sharpe ratios. The authors' companion procedure forms one minimum-variance aggregate series per cluster and recomputes the variance of those cluster Sharpe ratios.
- **Evidence:** `d4_search` computes `labels` but never uses them after line 177; every effective-K case receives `var_all` at lines 182-185. The authors' companion code uses `variance_of_the_clustered_trials(X, clusters)` before applying DSR. Current reported DSRs are all near one, so this does not reverse the stated result, but the “paper route” label is inaccurate for effective K.
- **Smallest fix:** Either implement clustered trial aggregation and its variance, or label these rows as an ad hoc K-only sensitivity.

### F17 — MINOR — dependency versions are not reproducibly pinned

- **Location:** `pyproject.toml:10-23`; no lock file.
- **What is wrong:** Core dependencies have lower bounds only. Optimizer, clustering, dataframe and plotting behavior can move across versions; registered Sharpe gates round only to two decimals.
- **Evidence:** The fresh Python 3.14 environment resolved NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1 and matplotlib 3.11.2. Tests emitted a CVXPY “solution may be inaccurate” warning and a joblib CPU-detection warning. This particular resolution passed, but a future one is unconstrained.
- **Smallest fix:** Publish a lock/constraints file for the reference environment and test one minimum-supported environment separately.

### F18 — NIT — the requested install extras do not install the test runner

- **Location:** `pyproject.toml:19-23`.
- **What is wrong:** `pip install -e ".[notebooks,ref]"` does not include pytest, so the next `pytest` command cannot run in a clean venv. The README's own command uses `[dev,ref]` plus `jupyter` and does work; this defect is in the extras composition exposed by the project/setup request, not the README command.
- **Evidence:** Fresh environment: `.venv/bin/python -m pytest -q` returned `No module named pytest` until `pip install -e '.[dev]'`.
- **Smallest fix:** Add a combined `all`/`research` extra or document `[dev,notebooks,ref]` as the complete audit environment.

### F19 — NIT — prose/output checking is not automated in CI

- **Location:** all notebook Markdown; `README.md`; tests.
- **What is wrong:** Many exact numbers are hand-copied into Markdown. Existing tests pin selected values, but no test systematically reconciles prose with outputs.
- **Evidence:** I wrote `audit_tools/markdown_output_check.py` in the scratch clone. It extracted 3,516 numeric prose tokens; 624 lacked a literal textual match and 262 lacked a rounded numeric match in the same notebook/all notebook outputs. Manual triage showed these residuals were formulas, dates, design constants, cross-notebook references, or narrative approximations—not a confirmed precision mismatch—but the volume makes future drift likely.
- **Smallest fix:** Generate result prose/tables from named result objects where practical, and maintain a small machine-readable registry for headline values used by the README.

## Mutation checks

All mutations were made one at a time in `/private/tmp/maplab-review.VGGObb/mutations`, then restored. “Killed” means the named test failed as it should.

| # | Deliberate mutation | Targeted test | Result |
|---:|---|---|---|
| 1 | Newey–West lag always 0 | `test_robust.py::test_nw_lag_values` | **Killed** |
| 2 | Holm returns raw p-values | `test_robust.py::test_holm_bh_hand_examples` | **Killed** |
| 3 | Overlay uses same-day volatility (`shift(1)` removed) | `test_overlay.py::test_no_lookahead` | **Killed** |
| 4 | Same overlay look-ahead | `test_nb10_checks.py::test_overlay_identity` | **Survived** |
| 5 | Transaction cost changed from 10 bp to 0 | `test_nb08_checks.py::test_active_returns_vs_ew_match_the_quoted_values` | **Killed** (`-1.14` vs quoted `-1.36`) |
| 6 | Same zero-cost mutation | `test_nb01_03_checks.py::test_nb01_runs` | **Killed** (GMV Sharpe `0.65` vs `0.62`) |
| 7 | Data shape oracle changed 4611 → 4610 | `test_data_file.py::test_prices_shape_dates_columns` | **Killed** |
| 8 | DSR probability hard-coded to 0.5 | `test_sharpe.py::test_dsr_paper_example` | **Killed** (`0.5` vs `0.410`) |
| 9 | Same DSR mutation | `test_nb11_checks.py::test_build_data_matches_registered_sharpe` | **Survived** |
| 10 | Maximum drawdown hard-coded to 0 | `test_nb08_checks.py::test_drawdown_panel_titles_match_the_performance_table` | **Killed** |
| 11 | `Panel.slice` changed `< asof` to `<= asof` | two parametrizations of `test_models.py::test_no_lookahead` | **Killed both** |
| 12 | Backtest turnover doubled (remove 0.5 factor) | `test_models.py::test_fixed_weight_backtest_trades_drift_back` | **Survived** |
| 13 | Same doubled-turnover mutation | both `test_turnover_window.py` tests | **Survived both** |

The suite has useful layered reproduction gates, but tests 4, 9, 12 and 13 show that several files named in the requested review do not independently guard the mechanics their names suggest.

## Independent recomputation of headline numbers

I reconstructed net returns in the scratch clone and computed metrics with independent NumPy/Pandas formulas, including wealth starting at 1.0. At least five headline values reproduce:

| Run | Annual arithmetic return | Volatility | Sharpe vs BIL | MDD |
|---|---:|---:|---:|---:|
| EW | 7.0203% | 7.8743% | 0.7351 | -17.8151% |
| GMV(S) | 3.2024% | 3.1706% | 0.6232 | -6.6522% |
| MDP(S) | 3.8467% | 3.4640% | 0.7564 | -6.3741% |
| HRP(S) | 3.7204% | 3.8807% | 0.6429 | -9.2638% |
| MaxSharpe(S) | 5.7816% | 6.3116% | 0.7224 | -10.7996% |
| 60/40 | 10.4075% | 10.1063% | 0.9078 | -21.3056% |

These agree with README rounding. The DSR paper example (`SR*=0.036/0.079=0.4557`, `K=10`, trial variance `0.1`) also reproduces expected maximum `0.498`, maximum-SR standard deviation `0.186`, and DSR `0.410`.

## Prose-versus-output reconciliation

The extraction script scanned README and every Markdown cell against the newly executed notebook outputs. After normalization for rounding and percent-vs-decimal display, manual review of all residual Reading/TAKEAWAY/README contexts found **no confirmed numerical transcription mismatch at the stated precision**. The substantive unsupported/overstated claims are F01–F04, not copied-number errors.

## Things checked and found correct

- All strategy estimates reached through `Panel.slice` use rows strictly before the rebalance label. A `<`→`<=` mutation is caught for GMV and MaxSharpe.
- Calendar month-end labels are applied to returns from the first trading day at or after the label; the 2022-12-31 Saturday split therefore begins 2023 performance on 2023-01-03 without using that day's return in estimation.
- Estimation uses log returns; portfolio compounding uses simple returns.
- Newey–West lag rule `floor(4(T/100)^(2/9))` gives 9/8/6 for 4338/3504/834 observations.
- Delta-method Sharpe gradient, PSR, MinTRL and power formulas match the stated equations; PSR/MinTRL evaluate sampling variance at the null.
- Stationary-bootstrap indexing is seeded, geometrically distributed with mean block 21, shared across series, and bootstrap p-values are centered as stated.
- Holm and Benjamini–Hochberg implementations match hand examples; Romano–Wolf preserves step-down monotonicity and uses the registered fixed original Newey–West standard errors.
- Variance-ratio/Jensen identity `q_bar = 0.5 log(v_bar) - J` is algebraically correct; Spearman is used only descriptively.
- Jobson–Korkie/Memmel is not silently substituted: the repository explicitly uses a paired delta-method HAC test instead.
- Notebook 08–10 registrations precede their notebook commits. Notebook 11 is correctly labelled descriptive in the README/registration, although its integration tests are incomplete.
- Fresh resolution on Python 3.14 passed the complete suite and executed all 12 notebooks; all reported README table values reproduced.
- The shipped parquet has the pinned SHA, `(4611, 17)` shape, ordered unique dates, required columns, and no NaNs after construction.
- No tracked `__pycache__`, `.pyc`, `.DS_Store`, notebook-checkpoint, venv, or test-cache artifacts were found.

## Exact validation commands and results

Scratch clone: `/private/tmp/maplab-review.VGGObb/repo`.

```text
git clone --branch polish --single-branch /Users/frasagui/Projects/maplab-phase3 /private/tmp/maplab-review.VGGObb/repo
python3 -m venv .venv
.venv/bin/pip install -e '.[notebooks,ref]'
.venv/bin/python -m pytest -q
```

The first pytest attempt failed exactly with `No module named pytest`. After:

```text
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

result:

```text
255 passed, 2 warnings in 182.39s
```

Notebook execution:

```text
cd notebooks
for nb in [01][0-9]_*.ipynb; do
  ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 "$nb" || exit 1
done
```

The sandboxed attempt failed because Jupyter could not create its local kernel socket; the same command with normal local permissions executed notebooks 00 through 11 successfully, with no notebook failure.

Numeric audit:

```text
.venv/bin/python audit_tools/markdown_output_check.py \
  --repo . \
  --csv /private/tmp/maplab-review.VGGObb/markdown_number_audit.csv
```

result:

```text
numeric prose tokens: 3516; no exact output match: 624;
no rounded numeric match: 262
```

Those are review candidates rather than 262 errors; manual classification is summarized above.
