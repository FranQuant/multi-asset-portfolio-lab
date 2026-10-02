"""Sharpe-ratio inference after López de Prado, Lipton & Zoonekynd (2026),
"How to Use the Sharpe Ratio" (ADIA Lab Research Paper 19): variance of the
estimator under skew / kurtosis / autocorrelation, PSR, MinTRL, power,
the False Strategy Theorem and the Deflated Sharpe Ratio.

Independent implementation of the paper's formulas. Every Sharpe ratio here
is PER OBSERVATION (daily when fed daily returns), never annualised;
annualising (x sqrt(252)) is for labels only. `kurt` is the Pearson
(non-excess) kurtosis, so a Normal has kurt = 3. Functions marked "OUR
adaptation" are not in the paper.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

_EULER_GAMMA = float(np.euler_gamma)


def sample_moments(x) -> dict:
    """Per-observation SR (mean / std, ddof=1), skewness (bias=False), Pearson
    kurtosis (fisher=False, bias=False), lag-1 autocorrelation, and T."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or len(a) < 4 or np.isnan(a).any():
        raise ValueError("sample_moments: need a 1-D NaN-free array with >= 4 observations")
    return {
        "sr": float(a.mean() / a.std(ddof=1)),
        "skew": float(stats.skew(a, bias=False)),
        "kurt": float(stats.kurtosis(a, fisher=False, bias=False)),
        "rho": float(np.corrcoef(a[1:], a[:-1])[0, 1]),
        "T": int(len(a)),
    }


def sharpe_variance(sr: float, T: int, skew: float = 0., kurt: float = 3., rho: float = 0.) -> float:
    """Asymptotic variance of the SR estimator (eq. 2-3 / 5):
    (1/T)[(1+ρ)/(1−ρ) − (1+ρ+ρ²)/(1−ρ²)·γ3·SR + (1+ρ²)/(1−ρ²)·(γ4−1)/4·SR²].
    ρ = 0 and Normal returns reduce to (1 + SR²/2)/T."""
    if not -1.0 < rho < 1.0:
        raise ValueError(f"sharpe_variance: |rho| must be < 1, got {rho}")
    v = ((1 + rho) / (1 - rho)
         - (1 + rho + rho ** 2) / (1 - rho ** 2) * skew * sr
         + (1 + rho ** 2) / (1 - rho ** 2) * (kurt - 1) / 4 * sr ** 2) / T
    if v <= 0:
        raise ValueError(f"sharpe_variance: non-positive variance {v} (sr={sr}, skew={skew}, kurt={kurt}, rho={rho})")
    return float(v)


def psr(sr: float, sr0: float, T: int, skew: float = 0., kurt: float = 3., rho: float = 0.) -> float:
    """Probabilistic Sharpe ratio (eq. 9): Φ((SR − SR0)/σ0), σ0² the variance
    evaluated at SR0 (eq. 5), i.e. 1 − p-value of H0: SR = SR0 vs SR > SR0."""
    s0 = math.sqrt(sharpe_variance(sr0, T, skew, kurt, rho))
    return float(stats.norm.cdf((sr - sr0) / s0))


def min_trl(sr: float, sr0: float, skew: float = 0., kurt: float = 3., rho: float = 0.,
            alpha: float = .05) -> float:
    """Minimum track-record length in observations (eq. 11):
    T such that PSR = 1 − α; variance at SR0 with T = 1. NaN if sr <= sr0."""
    if sr <= sr0:
        return float("nan")
    z = stats.norm.ppf(1 - alpha)
    return float(sharpe_variance(sr0, 1, skew, kurt, rho) * (z / (sr - sr0)) ** 2)


def critical_sr(sr0: float, T: int, skew: float = 0., kurt: float = 3., rho: float = 0.,
                alpha: float = .05) -> float:
    """Critical value of the one-sided test at level α (eq. 8):
    SR0 + Φ⁻¹(1−α)·σ0."""
    return float(sr0 + stats.norm.ppf(1 - alpha) * math.sqrt(sharpe_variance(sr0, T, skew, kurt, rho)))


def power(sr0: float, sr1: float, T: int, skew: float = 0., kurt: float = 3., rho: float = 0.,
          alpha: float = .05) -> float:
    """Power 1 − β of H0: SR = SR0 vs H1: SR = SR1 (eq. 15-16):
    Φ((SR1 − SR_c)/σ1), σ1² the variance at SR1."""
    sc = critical_sr(sr0, T, skew, kurt, rho, alpha)
    s1 = math.sqrt(sharpe_variance(sr1, T, skew, kurt, rho))
    return float(stats.norm.cdf((sr1 - sc) / s1))


def t_for_power(sr0: float, sr1: float, target: float = .8, skew: float = 0., kurt: float = 3.,
                rho: float = 0., alpha: float = .05) -> int:
    """Smallest integer T in [10, 10^7] with power >= target (bisection;
    power is increasing in T for sr1 > sr0). Raises if 10^7 is not enough."""
    lo, hi = 10, 10 ** 7
    if power(sr0, sr1, lo, skew, kurt, rho, alpha) >= target:
        return lo
    if power(sr0, sr1, hi, skew, kurt, rho, alpha) < target:
        raise ValueError(f"t_for_power: power {target} not reached by T = {hi}")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power(sr0, sr1, mid, skew, kurt, rho, alpha) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def expected_max_sr(K: int, var: float, sr0: float = 0.) -> float:
    """Expected maximum SR of K independent trials with cross-trial variance
    `var` (False Strategy Theorem, eq. 28):
    SR0 + √var·[(1−γ)Φ⁻¹(1−1/K) + γΦ⁻¹(1−1/(Ke))], γ Euler–Mascheroni.
    K = 1 gives SR0 (the K -> 1 limit)."""
    if K < 1:
        raise ValueError(f"expected_max_sr: K must be >= 1, got {K}")
    if K == 1:
        return float(sr0)
    return float(sr0 + math.sqrt(var) * (
        (1 - _EULER_GAMMA) * stats.norm.ppf(1 - 1 / K)
        + _EULER_GAMMA * stats.norm.ppf(1 - 1 / (K * math.e))))


def sd_max_std_normal(K: int, n_nodes: int = 200) -> float:
    """Exact sd of the max of K iid N(0,1) (App. 4, eq. 62-65): moments of the
    density Kφ(x)Φ(x)^(K−1) by Gauss–Hermite quadrature with `n_nodes` nodes."""
    if K < 1:
        raise ValueError(f"sd_max_std_normal: K must be >= 1, got {K}")
    t, w = np.polynomial.hermite.hermgauss(n_nodes)
    x = math.sqrt(2.0) * t
    wt = w / math.sqrt(math.pi) * K * stats.norm.cdf(x) ** (K - 1)
    m1 = float(wt @ x)
    m2 = float(wt @ x ** 2)
    return math.sqrt(m2 - m1 ** 2)


def dsr(sr_star: float, sr_trials, K: int | None = None, var: float | None = None) -> dict:
    """Deflated Sharpe ratio, paper route (eq. 28-30): Φ((SR* − SR0K)/s0K) with
    SR0K = expected_max_sr(K, var), s0K = √var · sd_max_std_normal(K).
    `var` defaults to the sample variance (ddof=1) of `sr_trials`, `K` to
    len(sr_trials). Returns {K, var, sr0K, s0K, dsr}."""
    trials = np.asarray(sr_trials, dtype=float)
    if K is None:
        K = len(trials)
    if var is None:
        if len(trials) < 2:
            raise ValueError("dsr: need >= 2 trial SRs to estimate var")
        var = float(trials.var(ddof=1))
    sr0k = expected_max_sr(K, var)
    s0k = math.sqrt(var) * sd_max_std_normal(K)
    return {"K": int(K), "var": float(var), "sr0K": sr0k, "s0K": s0k,
            "dsr": float(stats.norm.cdf((sr_star - sr0k) / s0k))}


def effective_rank(C) -> float:
    """Effective number of trials of a correlation matrix (App. 3, third
    approach): exp of the entropy of its normalised positive eigenvalues."""
    ev = np.linalg.eigvalsh(np.asarray(C, dtype=float))
    p = ev[ev > 0]
    p = p / p.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def _silhouette_quality(sil: np.ndarray) -> float:
    """mean/std of the silhouette scores; -inf if the spread is zero or the
    ratio is not finite, so such a k is never selected."""
    sd = sil.std()
    if sd == 0:
        return -np.inf
    q = float(sil.mean() / sd)
    return q if np.isfinite(q) else -np.inf


def cluster_trials(C, max_k: int | None = None, n_init: int = 10, seed: int = 20261002):
    """Number of clusters of trials (App. 3, first approach): KMeans on the
    rows of D = √((1−C)/2), quality = mean/std of the silhouette scores, best
    over k = 2..max_k (default N−1). Seeded via random_state. A k whose quality
    is -inf (zero-spread silhouette) is never selected; raises if no k qualifies.
    Returns (k, labels)."""
    C = np.asarray(C, dtype=float)
    N = C.shape[0]
    D = np.sqrt((1 - np.clip(C, -1, 1)) / 2)
    max_k = N - 1 if max_k is None else min(max_k, N - 1)
    if max_k < 2:
        raise ValueError(f"cluster_trials: need N >= 3 trials, got {N}")
    best_q, best_k, best_labels = -np.inf, None, None
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(D)
        sil = silhouette_samples(D, labels)
        q = _silhouette_quality(sil)
        if q > best_q:
            best_q, best_k, best_labels = q, k, labels
    if best_k is None:
        raise ValueError("cluster_trials: no k with a finite silhouette quality")
    return best_k, best_labels


def sharpe_diff_power(delta: float, se: float, alpha: float = .05, n_tests: int = 1) -> float:
    """OUR adaptation (not in the paper): power of the two-sided test of a
    Sharpe difference `delta` with standard error `se`, z = Φ⁻¹(1 − α/(2·n_tests)):
    Φ(δ/se − z) + Φ(−δ/se − z). delta = 0 gives α/n_tests."""
    z = stats.norm.ppf(1 - alpha / (2 * n_tests))
    return float(stats.norm.cdf(delta / se - z) + stats.norm.cdf(-delta / se - z))


def years_for_power(delta: float, se: float, years: float, target: float = .8,
                    alpha: float = .05) -> float:
    """OUR adaptation: years of data for `sharpe_diff_power` to reach `target`,
    given that `se` was measured on `years` of data and se ∝ 1/√years."""
    if delta == 0:
        raise ValueError("years_for_power: delta = 0 has no finite solution")

    def gap(y: float) -> float:
        return sharpe_diff_power(delta, se * math.sqrt(years / y), alpha) - target

    lo, hi = years * 1e-6, years
    while gap(hi) < 0:
        hi *= 2
        if hi > 1e9 * years:
            raise ValueError("years_for_power: target not reachable")
    return float(optimize.brentq(gap, lo, hi, xtol=1e-12, rtol=1e-12))
