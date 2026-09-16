"""Tests for the ceiling on ``|z|``, and therefore the floor on ``p``.

The score test estimates its variance with the mean held at zero, so a
larger mean inflates the standard error along with the statistic. Two
bounds follow, and which one applies depends on whether the fitted ``rho``
is an interior optimum or has saturated ``rho_max``. The README states the
first one; both are checked here, because the weaker one is what the code
can actually reach and the stronger one is what a user meets.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sharp_cv import sharp_test
from sharp_cv._engine import SHA, SHARP, _fit_profile, _split_half_test

STRUCTURES = [SHARP, SHA]
IDS = [s.name for s in STRUCTURES]
# sqrt(J+1) for SHARP, sqrt(2) for SHA: the interior bound of each pattern.
INTERIOR = {"sharp": lambda n: np.sqrt(n + 1.0), "sha": lambda n: np.sqrt(2.0)}
# sqrt(2J) / sqrt(2K): what survives once rho is pinned at rho_max.
SATURATED = {"sharp": lambda n: np.sqrt(2.0 * n), "sha": lambda n: np.sqrt(2.0 * n)}


def _draw(rng, structure, n, rho, sigma=1.0, mu=0.0):
    cov = (np.eye(2 * n) + structure.corr_pattern(n) * rho) * sigma**2
    flat = mu + np.linalg.cholesky(cov) @ rng.standard_normal(2 * n)
    return flat.reshape(2, n).T


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
def test_interior_fits_obey_the_tighter_bound(structure):
    """``|z| < sqrt(J+1)`` (SHARP) / ``sqrt(2)`` (SHA) whenever the
    null-restricted ``rho`` is strictly inside its range. This is the bound
    the stationarity condition gives, and the one behind the documented
    p-value floor."""
    rng = np.random.default_rng(11)
    interior_seen = 0
    for n in (4, 5, 10, 30):
        for mu in (0.0, 1.0, 5.0, 50.0, 1e4):
            for _ in range(40):
                diff = _draw(rng, structure, n, rng.uniform(0.0, 0.45), 1.0, mu)
                _, rho, _ = _fit_profile(structure.spectrum(diff, 0.0),
                                         structure.rho_max)
                z = _split_half_test(diff, None, "st", structure).statistic
                if rho >= structure.rho_max * (1 - 1e-9):
                    continue            # saturated; only the weaker bound holds
                interior_seen += 1
                assert abs(z) <= INTERIOR[structure.name](n) * (1 + 1e-6), (
                    f"n={n} mu={mu} rho={rho}: |z|={abs(z)}"
                )
    assert interior_seen > 100, "the interior branch was barely exercised"


@pytest.mark.parametrize("structure", STRUCTURES, ids=IDS)
def test_saturated_fits_still_obey_the_weaker_bound(structure):
    """``|z| <= sqrt(2n)`` holds for every fit, interior or not: it follows
    from the eigen-decomposition alone, with no stationarity needed."""
    rng = np.random.default_rng(12)
    for n in (3, 8, 25):
        for mu in (0.0, 10.0, 1e3, 1e8):
            for _ in range(30):
                diff = _draw(rng, structure, n, rng.uniform(0.0, 0.45), 1.0, mu)
                z = _split_half_test(diff, None, "st", structure).statistic
                assert abs(z) <= SATURATED[structure.name](n) * (1 + 1e-9)


def test_pvalue_floor_is_reported_correctly_for_sharp():
    """The documented floor. An enormous difference at J = 5 still cannot
    be called significant at 0.01."""
    rng = np.random.default_rng(13)
    for J, floor in ((5, 1.431e-2), (10, 9.109e-4), (30, 2.577e-8)):
        assert np.isclose(2 * stats.norm.sf(np.sqrt(J + 1)), floor, rtol=1e-3)
    for _ in range(50):
        diff = _draw(rng, SHARP, 5, 0.2, 1.0, 20.0)
        assert sharp_test(diff, mode="st")[1] > 2 * stats.norm.sf(np.sqrt(2 * 5))
    # the strong form, on the fits that stay interior
    kept = 0
    for _ in range(200):
        diff = _draw(rng, SHARP, 5, 0.2, 1.0, 20.0)
        _, rho, _ = _fit_profile(SHARP.spectrum(diff, 0.0), SHARP.rho_max)
        if rho < SHARP.rho_max * (1 - 1e-9):
            kept += 1
            assert sharp_test(diff, mode="st")[1] > 0.01
    assert kept > 0


def test_saturation_needs_an_absurd_effect_size():
    """At J = 30 the fit stays interior well past any effect a bounded
    score could produce, so ``sqrt(J+1)`` is the ceiling in practice."""
    rng = np.random.default_rng(14)
    for mu in (1.0, 5.0, 20.0):
        saturated = 0
        for _ in range(200):
            diff = _draw(rng, SHARP, 30, 0.2, 1.0, mu)
            _, rho, _ = _fit_profile(SHARP.spectrum(diff, 0.0), SHARP.rho_max)
            saturated += rho >= SHARP.rho_max * (1 - 1e-9)
        assert saturated == 0, f"mu={mu} standard deviations already saturates rho"
