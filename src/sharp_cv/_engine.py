"""Shared engine behind :func:`sharp_test` and :func:`sha_test`.

Both tests take an ``[n, 2]`` array ``diff_AB`` of paired performance
differences (model 1 minus model 2). The ``AB`` in the name refers to the
two halves of the data, not to the two models: column 0 holds the values
obtained in half A, column 1 the values obtained in half B, and each row
pairs the two values that came from the same split of the data.

The two tests share every estimator. They differ only in the correlation
pattern they assume among the ``2n`` values, and therefore in the variance
of the grand mean:

* **SHARP**: the halves are redrawn on every repetition. The two values of
  one repetition come from disjoint halves and are uncorrelated. Any two
  values from different repetitions share data and are correlated at
  ``rho``, whether or not they come from the same half.
* **SHA**: the halves are drawn once and a single K-fold CV is run inside
  each half. Values within a half are correlated at ``rho``; values from
  different halves are uncorrelated.

Under either model the grand mean of ``diff_AB`` is the best linear
unbiased estimate of the true difference. ``mode`` selects how ``sigma^2``
and ``rho`` are estimated and how the statistic is formed:

``'mm'``
    Method of moments. ``sigma^2`` comes from the mean squared
    between-half difference, ``rho`` from the pooled within-half sample
    variance. If the implied variance of the mean falls below the naive
    independent-samples variance, ``rho`` is replaced by
    ``fall_back_rho``.
``'mmc'``
    As ``'mm'`` with ``rho`` clipped to ``[0, rho_clip]`` before use, where
    ``rho_clip`` is just inside the admissible range of the correlation
    pattern (0.497 for SHARP, 0.997 for SHA).
``'ml'``
    Gaussian maximum likelihood for ``sigma^2`` and ``rho`` with the mean
    fixed at the sample mean, followed by a Wald z-test.
``'rml'``
    Restricted maximum likelihood (likelihood of the mean-free contrasts),
    followed by a Wald z-test.
``'lrt'``
    Likelihood ratio test of "mean = 0" against "mean = sample mean". The
    signed square root of the statistic is reported as z.
``'st'``
    Score test: a Wald z-test whose variance uses ``sigma^2`` and ``rho``
    estimated under the null hypothesis of zero mean. Default.

The four likelihood-based modes parameterise ``rho`` as
``tanh(r**2) * rho_max`` and therefore confine it to ``[0, rho_max)``: they
never return a negative correlation. ``rho_max`` is set per pattern by the
positive-definiteness limit of that pattern's covariance -- 0.499 for
SHARP, whose covariance is singular at ``rho = 0.5``, and 0.999 for SHA,
whose block-diagonal covariance stays positive definite up to ``rho = 1``.
Only ``'mm'`` leaves ``rho`` unconstrained.

The fallback rule of ``'mm'``, the ``'mmc'`` mode and the bound on ``rho``
are implementation safeguards. The accompanying paper describes the five
estimators without them; its results used ``'st'``, which the fallback
rule never touches.

SHA is switched off in this release: :func:`sha_test` raises and is not
exported from :mod:`sharp_cv`; the SHA path is reachable only through
:func:`_split_half_test` with ``SHA``. Its pattern, variance of the mean
and bound on ``rho`` are kept intact so the test can return once its
estimator is replaced. See ``_SHA_MSG``.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from scipy.linalg import block_diag, cholesky, toeplitz

VALID_MODES = ("mm", "mmc", "ml", "rml", "lrt", "st")
# The only modes that read fall_back_rho.
_FALLBACK_MODES = ("mm", "mmc")

# rho is parameterised as tanh(r**2) * rho_max, so rho_max is the largest
# correlation the likelihood-based modes can reach. It differs by pattern
# because the two covariances lose positive definiteness at different
# points: the SHARP covariance is singular at rho = 0.5 (the contrast
# e_j + e_{j+n} - e_k - e_{k+n} has eigenvalue sigma^2 * (1 - 2 * rho)),
# while the block-diagonal SHA covariance has eigenvalues sigma^2 * (1 - rho)
# and sigma^2 * (1 + (n - 1) * rho), so it stays positive definite up to
# rho = 1. Capping SHA at the SHARP limit would saturate rho whenever the
# within-half fold correlation exceeds 0.5 and inflate the false-positive
# rate. _RHO_MARGIN keeps rho_clip strictly inside the range, so that
# arctanh(rho_clip / rho_max) stays finite.
_RHO_MARGIN = 0.002
_SHARP_RHO_MAX = 0.499
_SHA_RHO_MAX = 0.999

# SHA is closed off at its public entry points. A single split leaves only
# two half results, whatever K is, and the variance of their mean has to be
# estimated from those same two numbers, so it rises with the difference
# being tested and the two cancel: the test almost never rejects. The
# covariance code below is correct and stays in place; setting this to True
# re-opens the entry points.
_SHA_AVAILABLE = False
_SHA_MSG = (
    "The SHA test is not available in this release. A single split gives "
    "only two half results however many folds are used, which leaves the "
    "test too little to estimate its own uncertainty from, and in practice "
    "it almost never rejects. Use the SHARP test instead: pass a repeated "
    "splitter such as RepeatedKFold(n_splits=5, n_repeats=30) to "
    "sharp_cross_val_test, or call sharp_test on one row per repetition of "
    "the split-half procedure."
)


def _require_sha(context: str = "") -> None:
    """Raise unless SHA has been switched back on.

    ``context`` prefixes the message when the caller did not name SHA
    itself, so that e.g. ``cv=5`` explains why SHA came into it.
    """
    if not _SHA_AVAILABLE:
        raise NotImplementedError(context + _SHA_MSG)


class Structure(NamedTuple):
    """Correlation pattern of one split-half design.

    ``corr_pattern(n)`` returns a ``[2n, 2n]`` 0/1 matrix marking the
    entries of the correlation matrix that equal ``rho`` (the diagonal is
    zero). ``var_of_mean(n, sigma2, rho)`` is the variance of the grand
    mean of the ``2n`` values under that pattern. ``rho_max`` is the
    positive-definiteness limit of that covariance and bounds the
    likelihood-based estimates of ``rho``; ``rho_clip`` is the bound
    ``'mmc'`` clips to, one ``_RHO_MARGIN`` inside ``rho_max``.
    """

    name: str
    corr_pattern: Callable[[int], np.ndarray]
    var_of_mean: Callable[[int, float, float], float]
    rho_max: float

    @property
    def rho_clip(self) -> float:
        return self.rho_max - _RHO_MARGIN


def _sharp_pattern(n: int) -> np.ndarray:
    # Row i is uncorrelated with itself and with its partner in the other
    # half (same repetition), and correlated with everything else.
    return toeplitz(np.concatenate(([0], np.ones(n - 1), [0], np.ones(n - 1))))


def _sharp_var_of_mean(n: int, sigma2: float, rho: float) -> float:
    # 2n values; each is correlated with 2(n - 1) others.
    return sigma2 * (1 / (2 * n) + (n - 1) / n * rho)


def _sha_pattern(n: int) -> np.ndarray:
    # Correlation only inside each half; cross-half block is zero.
    off = np.ones((n, n)) - np.eye(n)
    return block_diag(off, off)


def _sha_var_of_mean(n: int, sigma2: float, rho: float) -> float:
    # 2n values; each is correlated with (n - 1) others.
    return sigma2 * (1.0 / (2 * n) + (n - 1) / (2 * n) * rho)


SHARP = Structure("sharp", _sharp_pattern, _sharp_var_of_mean, _SHARP_RHO_MAX)
SHA = Structure("sha", _sha_pattern, _sha_var_of_mean, _SHA_RHO_MAX)


class Fit(NamedTuple):
    """Full output of one test: statistic, p-value, mean and its standard
    error. ``mean`` and ``se`` are both NaN for ``'lrt'`` and ``'st'``,
    which do not form the statistic as a mean over a standard error."""

    statistic: float
    pvalue: float
    mean: float
    se: float


def sharp_test(diff_AB, fall_back_rho=None, mode: str = "st"):
    """SHARP test for paired split-half differences.

    Use when the split-half procedure was repeated: on every repetition
    the data were divided into fresh halves, a cross-validation was run
    inside each half, and the fold results of each half were averaged.
    Around 30 repetitions or more is a reasonable starting point when
    the inner scheme is K-fold, or around 150 or more when each half
    contributes a single Monte-Carlo train/test split.

    Args:
        diff_AB: Array of shape ``[J, 2]``. Row ``j`` holds the mean
            performance difference (model 1 minus model 2) in half A and
            in half B of repetition ``j``.
        fall_back_rho: Correlation used by ``'mm'`` and ``'mmc'`` when the
            estimated variance of the mean falls below its independent-
            samples minimum. Required for those two modes and ignored by
            every other mode, including the default, so it may be left as
            ``None``. Use ``1 / (2 * K)`` when a K-fold CV was run inside
            each half, or ``test_size / 2`` for a single Monte-Carlo split
            inside each half; both are the fraction of the *full* dataset
            held out by one inner test set, the heuristic used in the
            paper's simulations.
        mode: One of ``'mm'``, ``'mmc'``, ``'ml'``, ``'rml'``, ``'lrt'``,
            ``'st'`` (default). See the module docstring.

    Returns:
        ``(z, p)`` with a two-sided p-value. Both are NaN when fewer than
        two rows are given or the two halves are identical.
    """
    fit = _split_half_test(diff_AB, fall_back_rho, mode, SHARP)
    return fit.statistic, fit.pvalue


def sha_test(diff_AB, fall_back_rho=None, mode: str = "st"):
    """SHA test for paired split-half differences from a single K-fold run.

    **Not available in this release: this function always raises.** SHA
    divides the data into two halves once and runs a single K-fold CV
    inside each half. That leaves only two half results however many folds
    are used, too little for the test to estimate its own uncertainty
    from, and in practice it almost never rejects.

    Use :func:`sharp_test`, which takes one row per repetition of the
    split-half procedure with the halves redrawn every time. The signature
    is kept for when a working estimator replaces this one.

    Args:
        diff_AB: Array of shape ``[K, 2]``. Row ``k`` holds the
            performance difference (model 1 minus model 2) on fold ``k``
            of half A and on fold ``k`` of half B.
        fall_back_rho: As for :func:`sharp_test`.
        mode: As for :func:`sharp_test`.

    Raises:
        NotImplementedError: always, while SHA is switched off.
    """
    _require_sha()
    fit = _split_half_test(diff_AB, fall_back_rho, mode, SHA)
    return fit.statistic, fit.pvalue


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _validate_mode(mode) -> None:
    if mode == "all":
        raise ValueError("mode='all' is not supported; call the test once per mode. "
                         f"mode must be one of {VALID_MODES}.")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")


def _as_diff_AB(diff_AB) -> np.ndarray:
    diff = np.asarray(diff_AB, dtype=float)
    if diff.ndim != 2 or diff.shape[1] != 2:
        raise ValueError(f"diff_AB must have shape [n, 2]; got {diff.shape}")
    return diff


def _check_fall_back_rho(fall_back_rho, mode: str) -> float | None:
    """Return ``fall_back_rho`` as a float, or ``None`` when the mode does
    not read it. Only ``'mm'`` and ``'mmc'`` require a value."""
    if fall_back_rho is None:
        if mode in _FALLBACK_MODES:
            raise ValueError(
                f"fall_back_rho is required for mode={mode!r}. Use 1 / (2 * K) "
                "when a K-fold CV was run inside each half, or test_size / 2 "
                "for a single Monte-Carlo split inside each half."
            )
        return None
    fall_back_rho = float(fall_back_rho)
    if not np.isfinite(fall_back_rho):
        raise ValueError("fall_back_rho must be a finite number")
    return fall_back_rho


def _two_sided_p(z: float) -> float:
    return stats.norm.cdf(-abs(z)) * 2


def _logmvnpdf(y, mu, cov):
    """Gaussian log-density up to an additive constant."""
    L = cholesky(cov, lower=True)
    log_det = np.sum(np.log(np.diag(L)))
    diff = y - mu
    exponent = -0.5 * np.dot(diff.T, np.linalg.solve(cov, diff))
    return -log_det + exponent


def _minimize_from(nll, x0):
    """Nelder-Mead from ``x0``, then BFGS polish. Falls back to the best
    point seen if an optimiser fails."""
    try:
        x1 = opt.fmin(nll, x0, disp=False)
    except Exception:
        x1 = x0
    try:
        res = opt.minimize(
            nll,
            x1,
            method="BFGS",
            options={
                "gtol": 1e-4,
                "maxiter": 100,
                "disp": False
            },
        )
        if res.success:
            return res.x, res.fun
        return x1, nll(x1)
    except Exception:
        return x1, nll(x1)


def _minimize(nll, *starts):
    """Minimise ``nll`` from every start in ``starts``; keep the lowest point.

    More than one start is needed because ``rho`` is parameterised as
    ``tanh(r**2) * rho_max``, whose derivative in ``r`` vanishes at
    ``r = 0``. A start with ``rho`` near zero therefore sits on a flat
    ridge that a gradient method reports as converged, and the maximum
    likelihood fit drives ``rho`` to zero often enough that ``'rml'``,
    which starts from that fit, would otherwise stop on the ridge instead
    of at the optimum of the restricted likelihood.
    """
    results = [_minimize_from(nll, x0) for x0 in starts]
    finite = [(x, f) for x, f in results if np.isfinite(f)]
    if not finite:
        return results[0]
    return min(finite, key=lambda xf: xf[1])


def _split_half_test(diff_AB, fall_back_rho, mode: str, structure: Structure) -> Fit:
    """Run one split-half test and return the full :class:`Fit`."""
    _validate_mode(mode)
    diff_AB = _as_diff_AB(diff_AB)
    fall_back_rho = _check_fall_back_rho(fall_back_rho, mode)

    n = diff_AB.shape[0]
    nan = float("nan")
    if n < 2:
        return Fit(nan, nan, nan, nan)

    d_flat = diff_AB.flatten(order="F")
    d_A, d_B = diff_AB[:, 0], diff_AB[:, 1]
    pattern = structure.corr_pattern(n)
    eye = np.eye(2 * n)
    var_of_mean = structure.var_of_mean
    rho_max = structure.rho_max

    def rho_of(r):
        return np.tanh(r**2) * rho_max

    def loglik(mu, sig, r, y):
        cov = sig**2 * (eye + pattern * np.tanh(r**2) * rho_max)
        return _logmvnpdf(y, mu, cov)

    # Naive variance of the mean if all 2n values were independent. The MoM
    # estimate is not allowed to fall below it.
    var_min = np.var(diff_AB, ddof=1) / (2 * n)

    mu_hat = np.mean(diff_AB)
    sig2_mm = np.mean((d_A - d_B)**2) / 2
    if sig2_mm == 0:
        return Fit(nan, nan, nan, nan)
    s2_pooled = 0.5 * (np.var(d_A, ddof=1) + np.var(d_B, ddof=1))
    rho_mm = 1 - s2_pooled / sig2_mm

    def wald(sigma2, rho, use_fallback):
        var = var_of_mean(n, sigma2, rho)
        if use_fallback and var < var_min:
            var = var_of_mean(n, sigma2, fall_back_rho)
        se = np.sqrt(var)
        z = mu_hat / se
        return Fit(float(z), float(_two_sided_p(z)), float(mu_hat), float(se))

    if mode == "mm":
        return wald(sig2_mm, rho_mm, True)

    rho_mmc = np.clip(rho_mm, 0, structure.rho_clip)
    if mode == "mmc":
        return wald(sig2_mm, rho_mmc, True)

    # Likelihood-based modes start from the constrained MoM estimate.
    init = [max(1e-2, np.sqrt(sig2_mm)), np.sqrt(np.arctanh(rho_mmc / rho_max))]

    if mode in ("ml", "rml", "lrt"):
        theta_ml, nll_ml = _minimize(
            lambda x: -loglik(mu_hat, x[0], x[1], d_flat), init
        )
        sig2_ml = theta_ml[0]**2
        rho_ml = rho_of(theta_ml[1])
        if mode == "ml":
            return wald(sig2_ml, rho_ml, False)

    if mode == "rml":
        # Project onto the space orthogonal to the mean, drop one redundant row.
        R = np.eye(2 * n) - np.ones((2 * n, 2 * n)) / (2 * n)
        R = R[:-1, :]
        RPR = R @ pattern @ R.T
        RPR = (RPR + RPR.T) / 2
        RRt = R @ R.T

        def rloglik(sig, r, y):
            cov = sig**2 * (RRt + RPR * np.tanh(r**2) * rho_max)
            return _logmvnpdf(R @ y, 0, cov)

        # ``theta_ml`` usually has rho at zero, which is a flat ridge of this
        # objective; ``init`` is the same starting point the other modes use
        # and is off the ridge whenever the moment estimate of rho is
        # positive. Trying both and keeping the lower is never worse.
        theta_rml, _ = _minimize(
            lambda x: -rloglik(x[0], x[1], d_flat),
            theta_ml,
            init,
        )
        sig2_rml = theta_rml[0]**2
        rho_rml = rho_of(theta_rml[1])
        return wald(sig2_rml, rho_rml, False)

    # 'lrt' and 'st' need the fit under the null hypothesis mean = 0.
    theta_0, nll_0 = _minimize(lambda x: -loglik(0, x[0], x[1], d_flat), init)

    if mode == "lrt":
        x2 = 2 * (nll_0 - nll_ml)
        if x2 <= 0:
            x2 = 0
        z = np.sign(mu_hat) * np.sqrt(x2)
        return Fit(float(z), float(_two_sided_p(z)), nan, nan)

    # 'st'
    sig2_0 = theta_0[0]**2
    rho_0 = rho_of(theta_0[1])
    z = mu_hat / np.sqrt(var_of_mean(n, sig2_0, rho_0))
    return Fit(float(z), float(_two_sided_p(z)), nan, nan)
