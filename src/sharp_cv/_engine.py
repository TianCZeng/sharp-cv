"""Statistics behind :func:`sharp_test` and :func:`sharp_confint`.

Both take an ``[n, 2]`` array ``diff_AB`` of paired performance
differences (model 1 minus model 2). ``A`` and ``B`` are the two halves of
the data, not the two models: column 0 holds the values from half A,
column 1 those from half B, and each row comes from one split of the data.

The test assumes a correlation pattern among the ``2n`` values:

* **SHARP**: the halves are redrawn on every repetition. The two values of
  one repetition come from disjoint halves and are uncorrelated. Any two
  values from different repetitions share data and are correlated at
  ``rho``, whether or not they come from the same half.
* **SHA** (not available in this release): the halves are drawn once and
  a single K-fold CV is run inside each half. Values within a half are
  correlated at ``rho``; values from different halves are uncorrelated.

The grand mean of ``diff_AB`` estimates the true difference. ``mode``
selects how the variance ``sigma^2`` and the correlation ``rho`` are
estimated and how the statistic is formed:

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

The four likelihood-based modes keep ``rho`` in ``[0, rho_max]``, so they
never return a negative correlation. ``rho_max`` sits just inside the
point where the covariance stops being positive definite: 0.499 for
SHARP (singular at ``rho = 0.5``) and 0.999 for SHA. Only ``'mm'`` leaves
``rho`` unconstrained.

The fallback rule of ``'mm'``, the ``'mmc'`` mode and the bound on ``rho``
are safeguards added in this package. The paper describes the five
estimators without them; its results used ``'st'``, which the fallback
rule never touches.

How the likelihood is fitted
----------------------------

Both correlation matrices have three distinct eigenvalues, each a linear
function of ``rho``, so the likelihood has a closed form and no
``2n x 2n`` matrix is built (see :class:`Eigenspace`). For each ``rho``
the best ``sigma^2`` is also closed form, which leaves a search over
``rho`` alone. For the SHARP ``'rml'`` fit the best ``rho`` is a formula
(:func:`_sharp_reml`). The other fits evaluate ``rho`` on a grid over
``[0, rho_max]`` and refine the best grid point with a bounded search.

SHA is switched off in this release: :func:`sha_test` always raises and
is not exported from :mod:`sharp_cv`. Its code is kept so that it can be
switched back on with ``_SHA_AVAILABLE``.
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

# Largest rho the likelihood-based modes can return. The SHARP covariance
# is singular at rho = 0.5 (eigenvalue 1 - 2 * rho), the SHA covariance only
# at rho = 1 (eigenvalue 1 - rho). Using the SHARP bound for SHA would cap
# rho too low and inflate its false-positive rate. 'mmc' clips one
# _RHO_MARGIN further inside.
_RHO_MARGIN = 0.002
_SHARP_RHO_MAX = 0.499
_SHA_RHO_MAX = 0.999

# SHA is switched off: with a single split the test almost never rejects
# (see _SHA_MSG). Setting this to True switches it back on.
_SHA_AVAILABLE = False
_SHA_MSG = ("The SHA test is not available in this release. A single split gives "
            "only two half results however many folds are used, which leaves the "
            "test too little to estimate its own uncertainty from, and in practice "
            "it almost never rejects. Use the SHARP test instead: pass a repeated "
            "splitter such as RepeatedKFold(n_splits=5, n_repeats=30) to "
            "sharp_cross_val_test, or call sharp_test on one row per repetition of "
            "the split-half procedure.")


def _require_sha(context: str = "") -> None:
    """Raise ``NotImplementedError`` while SHA is switched off.

    ``context`` is prepended to the message.
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


def _sharp_reml(spaces: tuple[Eigenspace, ...], rho_max: float):
    """Restricted maximum likelihood fit ``(sigma2, rho)``, in closed form.

    ``spaces`` is :func:`_sharp_spectrum` at any mean; the mean direction
    is dropped. Write ``Q2`` and ``Q3`` for the ``ss`` of the ``s`` and
    ``t`` eigenspaces. For each ``rho``, the best ``sigma^2`` is
    ``(Q2 / (1 - 2 rho) + Q3) / (2n - 1)``, and the profile deviance
    decreases below ``rho* = (1 - n Q2 / ((n - 1) Q3)) / 2`` and increases
    above it. The fit is therefore ``rho*`` clipped to ``[0, rho_max]``;
    ``Q2 = 0`` gives ``rho* = 1/2`` and so ``rho_max``. At an interior fit
    ``sigma^2`` equals the moment estimate ``Q3 / n`` and the variance of
    the mean is ``(Q3 - Q2) / (2n)``.

    Requires ``Q3 > 0`` (the two halves differ), which :func:`_fit` checks
    before fitting.
    """
    _, s_space, t_space = spaces
    n = t_space.mult
    rho = 0.5 * (1.0 - n * s_space.ss / ((n - 1) * t_space.ss))
    rho = min(max(rho, 0.0), rho_max)
    sigma2 = (s_space.ss / s_space.lam(rho) + t_space.ss) / (2 * n - 1)
    return sigma2, rho


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


def _sha_reml(spaces: tuple[Eigenspace, ...], rho_max: float):
    """Restricted maximum likelihood fit ``(sigma2, rho)`` by grid search."""
    sigma2, rho, _ = _fit_profile(spaces[1:], rho_max)
    return sigma2, rho


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
    ``reml(spaces, rho_max)`` takes that spectrum and returns the
    restricted maximum likelihood ``(sigma2, rho)``.
    ``corr_pattern`` is not used in fitting; it states the model the
    spectrum is derived from.
    """

    name: str
    corr_pattern: Callable[[int], np.ndarray]
    var_of_mean: Callable[[int, float, float], float]
    rho_max: float
    spectrum: Callable[[np.ndarray, float], tuple[Eigenspace, ...]]
    reml: Callable[[tuple[Eigenspace, ...], float], tuple[float, float]]

    @property
    def rho_clip(self) -> float:
        return self.rho_max - _RHO_MARGIN


SHARP = Structure(
    "sharp",
    _sharp_pattern,
    _sharp_var_of_mean,
    _SHARP_RHO_MAX,
    _sharp_spectrum,
    _sharp_reml,
)
SHA = Structure("sha", _sha_pattern, _sha_var_of_mean, _SHA_RHO_MAX, _sha_spectrum, _sha_reml)


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
    Around 30 repetitions is a reasonable starting point for K-fold
    inside each half, or around 150 for a single train/test split inside
    each half.

    Args:
        diff_AB: Array of shape ``[J, 2]``. Row ``j`` holds the mean
            performance difference (model 1 minus model 2) in half A and
            in half B of repetition ``j``.
        fall_back_rho: Only for ``'mm'`` and ``'mmc'``, which require it;
            ignored by the other modes. The correlation used when the
            estimated variance of the mean falls below the value for
            independent samples. Use ``1 / (2 * K)`` for K-fold inside
            each half, or ``test_size / 2`` for a single train/test split
            inside each half.
        mode: ``'st'`` (score test, default and recommended), ``'lrt'``
            (likelihood ratio test), ``'ml'`` or ``'rml'`` (maximum or
            restricted maximum likelihood), ``'mm'`` or ``'mmc'`` (method
            of moments, ``'mmc'`` with the correlation clipped). See
            :mod:`sharp_cv._engine` for details.

    Returns:
        ``(z, p)`` with a two-sided p-value. Both are NaN when fewer than
        two rows are given or the two halves are identical.
    """
    fit = _split_half_test(diff_AB, fall_back_rho, mode, SHARP)
    return fit.statistic, fit.pvalue


def sharp_confint(diff_AB, level: float = 0.95, fall_back_rho=None, mode: str = "st"):
    """Confidence interval for the true performance difference.

    The interval is the set of hypothesised differences the test does not
    reject at ``1 - level``, so it agrees with :func:`sharp_test`: it
    excludes 0 exactly when ``p < 1 - level``. For ``'ml'``, ``'rml'``,
    ``'mm'`` and ``'mmc'`` it is the usual ``mean +/- z * se``.

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

    **Not available in this release: this function always raises.** With
    a single split there are only two half results, however many folds
    are used, and in practice the test almost never rejects. Use
    :func:`sharp_test` instead.

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
# Likelihood: closed form, sigma^2 concentrated out, 1-D fit in rho
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
    mult = np.array([float(s.mult) for s in spaces])
    logdet = np.einsum("i,ij->j", mult, np.log(lam_safe))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = dof * np.log(q / dof) + logdet
    return np.where(ok & (q > 0), out, np.inf)


# Grid for the rho search. The evenly spaced points locate the best rho
# (except near a tie between two local minima, see the module docstring)
# and a bounded search then refines it. The extra points, packed towards
# rho_max, catch an optimum just below the bound, which happens when the
# tested mean is far from the sample mean.
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
        if mode == "ml":
            sigma2, rho, _ = _fit_profile(spaces, rho_max)
        else:
            sigma2, rho = structure.reml(spaces, rho_max)
        return wald(sigma2, rho, False)

    # 'lrt' and 'st' estimate the nuisance parameters under the hypothesis.
    spaces_0 = structure.spectrum(diff_AB, null_mean)
    sigma2_0, rho_0, dev_0 = _fit_profile(spaces_0, rho_max)

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
    sig2_mm = np.mean(np.diff(diff_AB, axis=1)**2) / 2
    step = float(np.sqrt(structure.var_of_mean(diff_AB.shape[0], sig2_mm, 0.0)))
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
