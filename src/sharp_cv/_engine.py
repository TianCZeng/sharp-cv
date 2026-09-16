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

The four likelihood-based modes confine ``rho`` to ``[0, rho_max]``: they
never return a negative correlation. ``rho_max`` is set per pattern by the
positive-definiteness limit of that pattern's covariance -- 0.499 for
SHARP, whose covariance is singular at ``rho = 0.5``, and 0.999 for SHA,
whose block-diagonal covariance stays positive definite up to ``rho = 1``.
Only ``'mm'`` leaves ``rho`` unconstrained.

The fallback rule of ``'mm'``, the ``'mmc'`` mode and the bound on ``rho``
are implementation safeguards. The accompanying paper describes the five
estimators without them; its results used ``'st'``, which the fallback
rule never touches.

How the likelihood is fitted
----------------------------

Both correlation matrices have only three distinct eigenvalues, each one
affine in ``rho``, so the log-likelihood can be written down in closed
form without ever building or factorising a ``2n x 2n`` matrix. ``sigma^2``
is then concentrated out analytically, which leaves an exact search in
``rho`` alone over a bounded interval. See :class:`Eigenspace`.

This replaced a two-dimensional Nelder-Mead/BFGS search over
``(sigma, r)`` with ``rho = tanh(r**2) * rho_max``, which was both slower
and less reliable. Slower because every evaluation factorised a dense
``2n x 2n`` covariance, so a fit cost ``O(n**3)``; the one-dimensional
search is 5x faster at ``n = 10`` and three orders of magnitude faster at
``n = 300``, because its cost does not grow with ``n`` at all. Less
reliable because the profile deviance need not be unimodal, so a local
optimiser could settle in the wrong basin, and because ``tanh(r**2)`` has
zero derivative at ``r = 0``, which makes ``r = 0`` a stationary point of
the reparameterised objective whatever the data say -- and the starting
point landed there whenever the moment estimate of ``rho`` was not
positive. Searching ``rho`` directly on a grid over the whole admissible
interval has neither defect, and the fit is checked against an exhaustive
scan in ``tests/test_likelihood.py``.

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
from scipy.linalg import block_diag, toeplitz

VALID_MODES = ("mm", "mmc", "ml", "rml", "lrt", "st")
# The only modes that read fall_back_rho.
_FALLBACK_MODES = ("mm", "mmc")
# Modes whose standard error does not depend on the mean being tested, so
# that inverting the test gives the plain Wald interval.
_WALD_MODES = ("mm", "mmc", "ml", "rml")

# rho is searched over [0, rho_max], so rho_max is the largest correlation
# the likelihood-based modes can reach. It differs by pattern because the
# two covariances lose positive definiteness at different points: the SHARP
# covariance is singular at rho = 0.5 (the contrast
# e_j + e_{j+n} - e_k - e_{k+n} has eigenvalue sigma^2 * (1 - 2 * rho)),
# while the block-diagonal SHA covariance has eigenvalues sigma^2 * (1 - rho)
# and sigma^2 * (1 + (n - 1) * rho), so it stays positive definite up to
# rho = 1. Capping SHA at the SHARP limit would saturate rho whenever the
# within-half fold correlation exceeds 0.5 and inflate the false-positive
# rate. _RHO_MARGIN keeps both rho_max and rho_clip strictly inside the
# range, so that every eigenvalue stays bounded away from zero.
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
_SHA_MSG = ("The SHA test is not available in this release. A single split gives "
            "only two half results however many folds are used, which leaves the "
            "test too little to estimate its own uncertainty from, and in practice "
            "it almost never rejects. Use the SHARP test instead: pass a repeated "
            "splitter such as RepeatedKFold(n_splits=5, n_repeats=30) to "
            "sharp_cross_val_test, or call sharp_test on one row per repetition of "
            "the split-half procedure.")


def _require_sha(context: str = "") -> None:
    """Raise unless SHA has been switched back on.

    ``context`` prefixes the message when the caller did not name SHA
    itself, so that e.g. ``cv=5`` explains why SHA came into it.
    """
    if not _SHA_AVAILABLE:
        raise NotImplementedError(context + _SHA_MSG)


# ---------------------------------------------------------------------------
# Correlation patterns
# ---------------------------------------------------------------------------


class Eigenspace(NamedTuple):
    """One eigenspace of the correlation matrix ``R(rho) = I + pattern * rho``.

    Both patterns have three of these, and in both the eigenvalue is affine
    in ``rho``: ``lam(rho) = intercept + slope * rho``. ``mult`` is the
    dimension of the eigenspace and ``ss`` the squared length of the
    projection of the data onto it, so that

    * ``log|R(rho)|`` is ``sum(mult * log(lam(rho)))`` and
    * ``y' R(rho)^-1 y`` is ``sum(ss / lam(rho))``

    with no matrix built. The mean direction is always listed first, and is
    the only one whose ``ss`` depends on the mean being tested; dropping it
    turns the likelihood into the restricted (ReML) likelihood.
    """

    intercept: float
    slope: float
    mult: int
    ss: float

    def lam(self, rho):
        return self.intercept + self.slope * rho


def _sharp_pattern(n: int) -> np.ndarray:
    # Row i is uncorrelated with itself and with its partner in the other
    # half (same repetition), and correlated with everything else.
    return toeplitz(np.concatenate(([0], np.ones(n - 1), [0], np.ones(n - 1))))


def _sharp_var_of_mean(n: int, sigma2: float, rho: float) -> float:
    # 2n values; each is correlated with 2(n - 1) others.
    return sigma2 * (1 / (2 * n) + (n - 1) / n * rho)


def _sharp_spectrum(diff_AB: np.ndarray, mean: float) -> tuple[Eigenspace, ...]:
    """Eigenspaces of the SHARP correlation matrix.

    With ``s = d_A + d_B`` and ``t = d_A - d_B``: the grand mean carries
    ``1 + 2(n-1) rho``, the ``s`` contrasts about their own mean carry
    ``1 - 2 rho`` and the ``t`` contrasts carry 1. The first eigenvalue
    over ``2n`` is :func:`_sharp_var_of_mean`; the other two give the
    admissible range of ``rho``.
    """
    n = diff_AB.shape[0]
    d_A, d_B = diff_AB[:, 0], diff_AB[:, 1]
    s = d_A + d_B
    centred = float(np.mean(diff_AB)) - mean
    return (
        Eigenspace(1.0, 2.0 * (n - 1), 1, 2 * n * centred**2),
        Eigenspace(1.0, -2.0, n - 1, 0.5 * float(np.sum((s - s.mean())**2))),
        Eigenspace(1.0, 0.0, n, 0.5 * float(np.sum((d_A - d_B)**2))),
    )


def _sha_pattern(n: int) -> np.ndarray:
    # Correlation only inside each half; cross-half block is zero.
    off = np.ones((n, n)) - np.eye(n)
    return block_diag(off, off)


def _sha_var_of_mean(n: int, sigma2: float, rho: float) -> float:
    # 2n values; each is correlated with (n - 1) others.
    return sigma2 * (1.0 / (2 * n) + (n - 1) / (2 * n) * rho)


def _sha_spectrum(diff_AB: np.ndarray, mean: float) -> tuple[Eigenspace, ...]:
    """Eigenspaces of the SHA correlation matrix.

    Each half is an equicorrelated block, so each contributes one
    ``1 + (n-1) rho`` direction (its own mean) and ``n - 1`` directions at
    ``1 - rho``. The two block means rotate into the grand mean and the
    between-half contrast, which share the eigenvalue and are listed
    separately only because the first depends on the mean being tested.
    """
    n = diff_AB.shape[0]
    d_A, d_B = diff_AB[:, 0], diff_AB[:, 1]
    within = float(np.sum((d_A - d_A.mean())**2) + np.sum((d_B - d_B.mean())**2))
    centred = float(np.mean(diff_AB)) - mean
    return (
        Eigenspace(1.0, n - 1.0, 1, 2 * n * centred**2),
        Eigenspace(1.0, n - 1.0, 1, 0.5 * n * (d_A.mean() - d_B.mean())**2),
        Eigenspace(1.0, -1.0, 2 * (n - 1), within),
    )


class Structure(NamedTuple):
    """Correlation pattern of one split-half design.

    ``corr_pattern(n)`` returns a ``[2n, 2n]`` 0/1 matrix marking the
    entries of the correlation matrix that equal ``rho`` (the diagonal is
    zero). ``var_of_mean(n, sigma2, rho)`` is the variance of the grand
    mean of the ``2n`` values under that pattern. ``rho_max`` is the
    positive-definiteness limit of that covariance and bounds the
    likelihood-based estimates of ``rho``; ``rho_clip`` is the bound
    ``'mmc'`` clips to, one ``_RHO_MARGIN`` inside ``rho_max``.
    ``spectrum(diff_AB, mean)`` is the eigen-decomposition the
    likelihood-based modes are fitted through; see :class:`Eigenspace`.
    ``corr_pattern`` is not used to fit anything and is kept because it
    states the model the spectrum is derived from.
    """

    name: str
    corr_pattern: Callable[[int], np.ndarray]
    var_of_mean: Callable[[int, float, float], float]
    rho_max: float
    spectrum: Callable[[np.ndarray, float], tuple[Eigenspace, ...]]

    @property
    def rho_clip(self) -> float:
        return self.rho_max - _RHO_MARGIN


SHARP = Structure("sharp", _sharp_pattern, _sharp_var_of_mean, _SHARP_RHO_MAX, _sharp_spectrum)
SHA = Structure("sha", _sha_pattern, _sha_var_of_mean, _SHA_RHO_MAX, _sha_spectrum)


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


def sharp_confint(diff_AB, level: float = 0.95, fall_back_rho=None, mode: str = "st"):
    """Confidence interval for the true performance difference.

    The interval is the set of hypothesised differences the test does not
    reject at ``1 - level``, so it agrees with :func:`sharp_test` by
    construction: the interval excludes 0 exactly when ``p < 1 - level``.
    For ``'st'`` and ``'lrt'`` the nuisance parameters are re-estimated at
    every hypothesised difference, which is what makes the two agree; for
    the Wald modes the standard error does not depend on the hypothesis and
    the interval reduces to ``mean +/- z * se``.

    An end is ``-inf`` or ``inf`` when no finite difference is rejected on
    that side at the requested ``level``.

    Args:
        diff_AB: as for :func:`sharp_test`.
        level: coverage, e.g. 0.95 for a 95% interval. Must lie in (0, 1).
        fall_back_rho: as for :func:`sharp_test`.
        mode: as for :func:`sharp_test`.

    Returns:
        ``(lo, hi)`` in the units of the score. Both are NaN when
        :func:`sharp_test` would return NaN.
    """
    return _split_half_confint(diff_AB, level, fall_back_rho, mode, SHARP)


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
            raise ValueError(f"fall_back_rho is required for mode={mode!r}. Use 1 / (2 * K) "
                             "when a K-fold CV was run inside each half, or test_size / 2 "
                             "for a single Monte-Carlo split inside each half.")
        return None
    fall_back_rho = float(fall_back_rho)
    if not np.isfinite(fall_back_rho):
        raise ValueError("fall_back_rho must be a finite number")
    return fall_back_rho


def _two_sided_p(z: float) -> float:
    return stats.norm.cdf(-abs(z)) * 2


# ---------------------------------------------------------------------------
# Likelihood: closed form, sigma^2 concentrated out, exact 1-D fit in rho
# ---------------------------------------------------------------------------


def _deviance(spaces: tuple[Eigenspace, ...], rho, dof: int):
    """Twice the profile negative log-likelihood, up to a constant.

    ``sigma^2`` is replaced by its maximiser ``q(rho) / dof``, leaving

        ``dof * log(q(rho) / dof) + sum(mult * log(lam(rho)))``

    where ``q(rho) = sum(ss / lam(rho))``. The dropped constant is
    ``dof * (1 + log(2 pi))`` and is the same for every ``spaces``, so
    differences of this function are differences of ``-2 * loglik``, which
    is what ``'lrt'`` needs. Vectorised over ``rho``; non-positive
    eigenvalues give ``inf``.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    lam = np.stack([s.lam(rho) for s in spaces])
    ok = np.all(lam > 0, axis=0)
    lam_safe = np.where(lam > 0, lam, 1.0)
    q = np.einsum("i,ij->j", np.array([s.ss for s in spaces]), 1.0 / lam_safe)
    logdet = np.einsum("i,ij->j", np.array([float(s.mult) for s in spaces]), np.log(lam_safe))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = dof * np.log(q / dof) + logdet
    return np.where(ok & (q > 0), out, np.inf)


# Grid resolution of the rho search. The objective is smooth on a bounded
# interval, so a grid this dense brackets the optimum and the bounded
# refinement below finds it to machine precision. The extra points hug
# rho_max geometrically: when the mean being tested is very large the
# optimum slides to within a tiny distance of the upper limit, which a
# uniform grid alone would step over.
_RHO_GRID = 257
_RHO_TAIL = np.logspace(-10.0, -2.0, 24)


def _fit_profile(spaces: tuple[Eigenspace, ...], rho_max: float):
    """Maximise the Gaussian likelihood over ``(sigma^2, rho >= 0)``.

    Returns ``(sigma2, rho, deviance)``. ``spaces`` fixes the model and the
    data; pass every eigenspace for the full likelihood, or all but the
    mean direction for the restricted one.
    """
    dof = sum(s.mult for s in spaces)
    grid = np.unique(
        np.clip(
            np.concatenate([np.linspace(0.0, rho_max, _RHO_GRID), rho_max * (1.0 - _RHO_TAIL)]),
            0.0,
            rho_max,
        ))
    values = _deviance(spaces, grid, dof)
    k = int(np.argmin(values))
    rho, best = float(grid[k]), float(values[k])
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    if hi > lo:
        res = opt.minimize_scalar(
            lambda r: float(_deviance(spaces, r, dof)[0]),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if res.fun <= best:
            rho, best = float(res.x), float(res.fun)
    sigma2 = float(sum(s.ss / s.lam(rho) for s in spaces) / dof)
    return sigma2, rho, best


def _split_half_test(diff_AB, fall_back_rho, mode: str, structure: Structure) -> Fit:
    """Run one split-half test and return the full :class:`Fit`."""
    _validate_mode(mode)
    diff_AB = _as_diff_AB(diff_AB)
    fall_back_rho = _check_fall_back_rho(fall_back_rho, mode)
    return _fit(diff_AB, fall_back_rho, mode, structure, null_mean=0.0)


def _fit(
    diff_AB: np.ndarray,
    fall_back_rho,
    mode: str,
    structure: Structure,
    null_mean: float,
) -> Fit:
    """One test of ``mean == null_mean``. Inputs are already validated.

    ``null_mean`` is 0 for the test itself and is swept by
    :func:`_split_half_confint`. Only ``'st'`` and ``'lrt'`` read it as
    anything other than a shift of the statistic: they re-estimate
    ``(sigma^2, rho)`` under the hypothesis, which the Wald modes do not.
    """
    n = diff_AB.shape[0]
    nan = float("nan")
    if n < 2:
        return Fit(nan, nan, nan, nan)

    d_A, d_B = diff_AB[:, 0], diff_AB[:, 1]
    var_of_mean = structure.var_of_mean
    rho_max = structure.rho_max

    # Naive variance of the mean if all 2n values were independent. The MoM
    # estimate is not allowed to fall below it.
    var_min = np.var(diff_AB, ddof=1) / (2 * n)

    mu_hat = float(np.mean(diff_AB))
    centred = mu_hat - null_mean
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
        z = centred / se
        return Fit(float(z), float(_two_sided_p(z)), float(mu_hat), float(se))

    if mode == "mm":
        return wald(sig2_mm, rho_mm, True)
    if mode == "mmc":
        return wald(sig2_mm, np.clip(rho_mm, 0, structure.rho_clip), True)

    if mode in ("ml", "rml"):
        # Both fit at the sample mean, so the mean direction carries no
        # residual; 'rml' drops that direction from the likelihood instead.
        spaces = structure.spectrum(diff_AB, mu_hat)
        sigma2, rho, _ = _fit_profile(spaces if mode == "ml" else spaces[1:], rho_max)
        return wald(sigma2, rho, False)

    # 'lrt' and 'st' estimate the nuisance parameters under the hypothesis.
    sigma2_0, rho_0, dev_0 = _fit_profile(structure.spectrum(diff_AB, null_mean), rho_max)

    if mode == "lrt":
        _, _, dev_ml = _fit_profile(structure.spectrum(diff_AB, mu_hat), rho_max)
        x2 = max(dev_0 - dev_ml, 0.0)
        z = np.sign(centred) * np.sqrt(x2)
        return Fit(float(z), float(_two_sided_p(z)), nan, nan)

    # 'st'
    z = centred / np.sqrt(var_of_mean(n, sigma2_0, rho_0))
    return Fit(float(z), float(_two_sided_p(z)), nan, nan)


# ---------------------------------------------------------------------------
# Confidence interval by inverting the test
# ---------------------------------------------------------------------------

# Bracketing budget for the root search. Each step doubles the distance
# from the point estimate, so 60 steps cover 1e18 standard errors: far
# enough that failing to bracket means the statistic has hit its ceiling
# rather than that the search was too short.
_BRACKET_STEPS = 60


def _split_half_confint(diff_AB, level, fall_back_rho, mode, structure: Structure):
    """Invert the test: the hypothesised means it does not reject."""
    _validate_mode(mode)
    diff_AB = _as_diff_AB(diff_AB)
    fall_back_rho = _check_fall_back_rho(fall_back_rho, mode)
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1); got {level}")

    at_zero = _fit(diff_AB, fall_back_rho, mode, structure, 0.0)
    if not np.isfinite(at_zero.statistic):
        return float("nan"), float("nan")

    mu_hat = float(np.mean(diff_AB))
    z_crit = float(stats.norm.isf((1.0 - level) / 2.0))

    if mode in _WALD_MODES:
        # se does not move with the hypothesis, so inversion is closed form.
        return mu_hat - z_crit * at_zero.se, mu_hat + z_crit * at_zero.se

    def excess(mu0: float) -> float:
        """How far ``|z|`` at ``mu0`` overshoots the critical value."""
        z = _fit(diff_AB, fall_back_rho, mode, structure, mu0).statistic
        return abs(z) - z_crit

    # A scale to step by: the standard error the moment estimator implies.
    step = float(np.sqrt(structure.var_of_mean(diff_AB.shape[0], np.mean(np.diff(diff_AB, axis=1)**2) / 2, 0.0)))
    if not np.isfinite(step) or step <= 0:
        step = max(abs(mu_hat), 1.0)
    return _root(excess, mu_hat, -step), _root(excess, mu_hat, step)


def _root(excess: Callable[[float], float], mu_hat: float, step: float) -> float:
    """First hypothesised mean on ``step``'s side of ``mu_hat`` that the
    test rejects, or an infinity when the statistic never gets there."""
    near = mu_hat
    for _ in range(_BRACKET_STEPS):
        far = mu_hat + step
        if excess(far) > 0:
            lo, hi = sorted((near, far))
            return float(opt.brentq(excess, lo, hi, xtol=1e-14, rtol=8.9e-16))
        near, step = far, step * 2.0
    return float("inf") if step > 0 else float("-inf")
