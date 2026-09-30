"""Tests for the closed-form likelihood and the one-dimensional fit.

The engine never builds the ``2n x 2n`` covariance: it works from the
three eigenspaces of the correlation matrix and concentrates ``sigma^2``
out, leaving a function of ``rho`` alone. These tests check that shortcut
against the thing it replaced -- the explicit matrix, decomposed here with
NumPy and not with any of the engine's own code. The SHARP fits need no
search over ``rho``; they are checked against an exhaustive scan of their
objective, the full fit also against a case with two near-tied minima,
and the restricted fit against the moment estimate it reduces to.
"""

from __future__ import annotations

import numpy as np
import pytest

from sharp_cv._engine import (
    SHA,
    SHARP,
    _deviance,
    _sharp_cubic,
    _split_half_test,
)

STRUCTURES = [SHARP, SHA]
IDS = [s.name for s in STRUCTURES]


def _draw(rng, structure, n, rho, sigma=1.0, mu=0.0):
    """One dataset straight from the structure's own covariance."""
    cov = (np.eye(2 * n) + structure.corr_pattern(n) * rho) * sigma**2
    flat = mu + np.linalg.cholesky(cov) @ rng.standard_normal(2 * n)
    return flat.reshape(2, n).T


def _nll_matrix(structure, diff, mean, sigma2, rho):
    """Gaussian negative log-likelihood from the explicit matrix, up to the
    same additive constant the engine drops."""
    n = diff.shape[0]
    cov = sigma2 * (np.eye(2 * n) + structure.corr_pattern(n) * rho)
    y = diff.flatten(order="F") - mean
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    return 0.5 * (logdet + y @ np.linalg.solve(cov, y))


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
@pytest.mark.parametrize("n", [2, 3, 5, 12])
def test_spectrum_matches_the_eigenvalues_of_the_pattern(structure, n):
    """The declared eigenvalues and multiplicities are the real ones, and
    ``rho_max`` sits just inside the range where all of them are positive.
    SHA's range ends at 1 and SHARP's at 0.5; capping SHA at SHARP's bound
    would understate its variance of the mean."""
    diff = np.zeros((n, 2))
    spaces = structure.spectrum(diff, 0.0)
    for rho in (0.0, 0.1, 0.37, structure.rho_max):
        matrix = np.eye(2 * n) + structure.corr_pattern(n) * rho
        declared = np.sort(
            np.concatenate([np.full(s.mult, s.lam(rho)) for s in spaces])
        )
        np.testing.assert_allclose(
            declared, np.sort(np.linalg.eigvalsh(matrix)), atol=1e-10
        )
    assert min(s.lam(structure.rho_max) for s in spaces) > 0
    assert min(s.lam(structure.rho_max + 0.002) for s in spaces) < 0


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
def test_var_of_mean_is_the_mean_direction_eigenvalue(structure):
    """``var_of_mean`` must be the first eigenvalue over ``2n``, or the
    statistic and the likelihood would be describing different models."""
    for n in (2, 4, 9, 40):
        mean_space = structure.spectrum(np.zeros((n, 2)), 0.0)[0]
        for sigma2 in (0.3, 1.0, 7.5):
            for rho in (0.0, 0.2, structure.rho_max):
                assert np.isclose(
                    structure.var_of_mean(n, sigma2, rho),
                    sigma2 * mean_space.lam(rho) / (2 * n),
                )


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
def test_deviance_matches_the_explicit_matrix_likelihood(structure):
    """``_deviance`` is ``-2 * loglik`` with ``sigma^2`` at its maximiser,
    up to a constant that does not depend on the data or on ``rho``."""
    rng = np.random.default_rng(101)
    for n in (3, 6, 20):
        for _ in range(5):
            diff = _draw(rng, structure, n, rng.uniform(0, 0.4), rng.uniform(0.3, 2))
            spaces = structure.spectrum(diff, 0.0)
            dim = sum(s.mult for s in spaces)
            offsets = []
            for rho in np.linspace(0.0, structure.rho_max, 9):
                dev = float(_deviance(spaces, rho)[0])
                sigma2 = sum(s.ss / s.lam(rho) for s in spaces) / dim
                offsets.append(2 * _nll_matrix(structure, diff, 0.0, sigma2, rho) - dev)
            # The offset is the same at every rho: the two agree as functions.
            np.testing.assert_allclose(offsets, offsets[0], rtol=1e-10)


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
def test_fits_find_the_global_optimum_of_their_objective(structure):
    """The full and the restricted fit are each checked against an
    exhaustive scan of the same objective; they must never be worse."""
    rng = np.random.default_rng(303)
    scan_grid = np.linspace(0.0, structure.rho_max, 8001)
    for n in (5, 12, 40):
        for _ in range(25):
            diff = _draw(
                rng,
                structure,
                n,
                rng.uniform(0, 0.45),
                rng.uniform(0.3, 2),
                rng.choice([0.0, 0.4]),
            )
            for mean in (0.0, float(np.mean(diff))):
                spaces = structure.spectrum(diff, mean)
                _, rho_full, dev_full = structure.fit(spaces, structure.rho_max)
                _, rho_reml = structure.reml(spaces, structure.rho_max)
                np.testing.assert_allclose(
                    dev_full, _deviance(spaces, rho_full)[0], rtol=1e-12
                )
                for included, rho in ((spaces, rho_full), (spaces[1:], rho_reml)):
                    dev = float(_deviance(included, rho)[0])
                    scan = float(np.min(_deviance(included, scan_grid)))
                    assert dev <= scan + 1e-9


def test_sharp_cubic_is_the_slope_of_the_deviance():
    """The slope of the SHARP profile deviance is ``4n P / (q lam1^2
    lam2^2)`` with ``P`` from :func:`_sharp_cubic`, so the roots of ``P``
    are exactly the points where the deviance can turn. The slope is
    differentiated here from the eigenspaces directly."""
    rng = np.random.default_rng(909)
    rho = np.linspace(0.0, SHARP.rho_max, 11)
    for n in (2, 3, 8, 40, 300):
        for _ in range(10):
            diff = _draw(
                rng,
                SHARP,
                n,
                rng.uniform(0, 0.45),
                rng.uniform(0.3, 2),
                rng.choice([0.0, 0.4]),
            )
            for mean in (0.0, float(np.mean(diff)), 3.0):
                spaces = SHARP.spectrum(diff, mean)
                dim = 2 * n
                q = sum(s.ss / s.lam(rho) for s in spaces)
                dq = -sum(s.ss * s.slope / s.lam(rho) ** 2 for s in spaces)
                dlogdet = sum(s.mult * s.slope / s.lam(rho) for s in spaces)
                slope = dim * dq / q + dlogdet
                lam1, lam2 = spaces[0].lam(rho), spaces[1].lam(rho)
                from_cubic = (
                    4 * n * np.polyval(_sharp_cubic(spaces), rho)
                    / (q * lam1**2 * lam2**2)
                )
                np.testing.assert_allclose(
                    from_cubic, slope, rtol=1e-9, atol=1e-9 * np.max(np.abs(slope))
                )


def test_sharp_fit_picks_the_lower_of_two_near_tied_minima():
    """Here the null-fit deviance has two local minima, near ``rho = 0.009``
    and ``rho = 0.138``, whose values differ by only 5e-5. A search that
    starts from grid points can settle on the higher one; comparing the
    candidates cannot."""
    n = 20
    template = SHARP.spectrum(np.zeros((n, 2)), 0.0)
    sums = (0.0337662, 0.542111, 1.0)
    spaces = tuple(s._replace(ss=q) for s, q in zip(template, sums))
    scan_grid = np.linspace(0.0, SHARP.rho_max, 400_001)
    scan = _deviance(spaces, scan_grid)
    inner = (scan[1:-1] < scan[:-2]) & (scan[1:-1] < scan[2:])
    minima, values = scan_grid[1:-1][inner], scan[1:-1][inner]
    assert minima.size == 2
    assert 0.0 < values[1] - values[0] < 1e-4  # the lower one is near 0.009
    _, rho, dev = SHARP.fit(spaces, SHARP.rho_max)
    assert abs(rho - minima[0]) <= scan_grid[1]
    assert dev <= values[0] + 1e-12


def test_sharp_reml_interior_fit_is_moment_matching():
    """At an interior ReML fit ``sigma^2`` is the moment estimate ``Q3 / n``
    and the variance of the mean is ``(Q3 - Q2) / (2n)``, where ``Q2`` is
    half the spread of the row sums and ``Q3`` half the squared row
    differences. The second is read from ``_split_half_test``, which also
    pins that ``'rml'`` goes through the closed form."""
    rng = np.random.default_rng(707)
    checked = 0
    for n in (3, 8, 30, 200):
        for _ in range(20):
            diff = _draw(rng, SHARP, n, rng.uniform(0.05, 0.4), rng.uniform(0.3, 2))
            s, t = diff[:, 0] + diff[:, 1], diff[:, 0] - diff[:, 1]
            q2, q3 = 0.5 * np.sum((s - s.mean()) ** 2), 0.5 * np.sum(t**2)
            sigma2, rho = SHARP.reml(SHARP.spectrum(diff, 0.0), SHARP.rho_max)
            if not 0.0 < rho < SHARP.rho_max:
                continue
            checked += 1
            np.testing.assert_allclose(sigma2, q3 / n, rtol=1e-12)
            fit = _split_half_test(diff, None, "rml", SHARP)
            np.testing.assert_allclose(fit.se**2, (q3 - q2) / (2 * n), rtol=1e-10)
    assert checked >= 50  # 67 of the 80 draws are interior with this seed
