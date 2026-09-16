"""Tests for the closed-form likelihood and the one-dimensional fit.

The engine never builds the ``2n x 2n`` covariance: it works from the
three eigenspaces of the correlation matrix, concentrates ``sigma^2`` out
and searches ``rho`` alone. These tests check that shortcut against the
thing it replaced -- the explicit matrix, decomposed here with NumPy and
not with any of the engine's own code.
"""

from __future__ import annotations

import numpy as np
import pytest

from sharp_cv._engine import SHA, SHARP, _deviance, _fit_profile

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
    """The declared eigenvalues and multiplicities are the real ones."""
    diff = np.zeros((n, 2))
    for rho in (0.0, 0.1, 0.37, structure.rho_max):
        matrix = np.eye(2 * n) + structure.corr_pattern(n) * rho
        declared = np.sort(
            np.concatenate(
                [np.full(s.mult, s.lam(rho)) for s in structure.spectrum(diff, 0.0)]
            )
        )
        np.testing.assert_allclose(
            declared, np.sort(np.linalg.eigvalsh(matrix)), atol=1e-10
        )


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
            dof = sum(s.mult for s in spaces)
            offsets = []
            for rho in np.linspace(0.0, structure.rho_max, 9):
                dev = float(_deviance(spaces, rho, dof)[0])
                sigma2 = sum(s.ss / s.lam(rho) for s in spaces) / dof
                offsets.append(2 * _nll_matrix(structure, diff, 0.0, sigma2, rho) - dev)
            # The offset is the same at every rho: the two agree as functions.
            np.testing.assert_allclose(offsets, offsets[0], rtol=1e-10)


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
def test_concentrated_sigma2_is_the_maximiser(structure):
    """At the fitted ``rho``, perturbing ``sigma^2`` cannot improve the
    explicit matrix likelihood."""
    rng = np.random.default_rng(202)
    for n in (4, 15):
        for _ in range(5):
            diff = _draw(rng, structure, n, rng.uniform(0, 0.4), rng.uniform(0.3, 2))
            sigma2, rho, _ = _fit_profile(
                structure.spectrum(diff, 0.0), structure.rho_max
            )
            best = _nll_matrix(structure, diff, 0.0, sigma2, rho)
            for factor in (0.9, 0.99, 1.01, 1.1):
                assert _nll_matrix(structure, diff, 0.0, sigma2 * factor, rho) >= best


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
def test_fit_finds_the_global_optimum_of_its_own_objective(structure):
    """The old two-dimensional search stalled on a flat ridge and missed
    its own optimum on a few percent of datasets. The one-dimensional
    search is checked here against an exhaustive scan of the same
    objective; it must never be worse."""
    rng = np.random.default_rng(303)
    scan_grid = np.linspace(0.0, structure.rho_max, 8001)
    for n in (5, 12, 40):
        for _ in range(25):
            diff = _draw(rng, structure, n, rng.uniform(0, 0.45), rng.uniform(0.3, 2),
                         rng.choice([0.0, 0.4]))
            for mean in (0.0, float(np.mean(diff))):
                spaces = structure.spectrum(diff, mean)
                for included in (spaces, spaces[1:]):   # full, then restricted
                    dof = sum(s.mult for s in included)
                    _, _, dev = _fit_profile(included, structure.rho_max)
                    scan = float(np.min(_deviance(included, scan_grid, dof)))
                    assert dev <= scan + 1e-9


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
def test_fit_stays_inside_its_bounds(structure):
    rng = np.random.default_rng(404)
    for n in (3, 10, 25):
        for mu in (0.0, 1.0, 1e4):
            diff = _draw(rng, structure, n, 0.3, 1.0, mu)
            for mean in (0.0, float(np.mean(diff))):
                sigma2, rho, _ = _fit_profile(structure.spectrum(diff, mean),
                                              structure.rho_max)
                assert 0.0 <= rho <= structure.rho_max
                assert sigma2 > 0.0


def test_deviance_rejects_inadmissible_rho():
    """Outside the positive-definite range the objective is ``inf``, so the
    search can never wander there."""
    rng = np.random.default_rng(505)
    diff = _draw(rng, SHARP, 8, 0.2)
    spaces = SHARP.spectrum(diff, 0.0)
    dof = sum(s.mult for s in spaces)
    bad = _deviance(spaces, np.array([0.5, 0.7, -0.2]), dof)
    assert np.all(np.isinf(bad))
