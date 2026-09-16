"""Tests for :func:`sharp_cv.sharp_confint`.

The interval is defined by inverting the test: it is the set of
hypothesised differences the test does not reject. That definition is what
these tests check, rather than any particular formula -- the interval has
to agree with the p-value, and its endpoints have to sit exactly where the
statistic crosses the critical value.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sharp_cv import VALID_MODES, sharp_confint, sharp_test
from sharp_cv._engine import SHARP, _fit, _split_half_confint

_Z95 = float(stats.norm.isf(0.025))


def _draw(rng, n, rho, sigma=1.0, mu=0.0):
    cov = (np.eye(2 * n) + SHARP.corr_pattern(n) * rho) * sigma**2
    flat = mu + np.linalg.cholesky(cov) @ rng.standard_normal(2 * n)
    return flat.reshape(2, n).T


@pytest.mark.parametrize("mode", VALID_MODES)
@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
def test_interval_agrees_with_the_pvalue(mode, level):
    """0 lies inside the interval exactly when the test does not reject.

    This is the property the interval exists to have, and the reason
    ``'st'`` and ``'lrt'`` re-estimate the nuisance parameters at every
    hypothesised mean instead of reusing the null fit.
    """
    rng = np.random.default_rng(7)
    for _ in range(120):
        n = int(rng.integers(4, 40))
        diff = _draw(rng, n, rng.uniform(0.0, 0.45), rng.uniform(0.2, 2.0),
                     rng.normal() * 0.4)
        p = sharp_test(diff, fall_back_rho=0.1, mode=mode)[1]
        lo, hi = sharp_confint(diff, level, fall_back_rho=0.1, mode=mode)
        assert (lo <= 0.0 <= hi) == (p >= 1 - level), (
            f"p={p} interval=({lo}, {hi})"
        )


@pytest.mark.parametrize("mode", ["st", "lrt"])
def test_endpoints_sit_on_the_critical_value(mode):
    """Just inside an endpoint the test accepts, just outside it rejects."""
    rng = np.random.default_rng(8)
    checked = 0
    for _ in range(40):
        n = int(rng.integers(6, 40))
        diff = _draw(rng, n, rng.uniform(0.0, 0.4), 1.0, rng.normal() * 0.5)
        mu_hat = float(np.mean(diff))
        lo, hi = sharp_confint(diff, 0.95, mode=mode)
        for end in (lo, hi):
            if not np.isfinite(end):
                continue
            outward = np.sign(end - mu_hat)
            step = max(abs(end - mu_hat) * 1e-5, 1e-10)
            z_in = _fit(diff, None, mode, SHARP, end - outward * step).statistic
            z_out = _fit(diff, None, mode, SHARP, end + outward * step).statistic
            assert abs(z_in) <= _Z95 * (1 + 1e-4)
            assert abs(z_out) >= _Z95 * (1 - 1e-4)
            checked += 1
    assert checked > 20


@pytest.mark.parametrize("mode", ["mm", "mmc", "ml", "rml"])
def test_wald_modes_reduce_to_mean_plus_minus_z_se(mode):
    """Their standard error does not move with the hypothesis, so
    inversion has a closed form and must match it exactly."""
    rng = np.random.default_rng(9)
    for _ in range(30):
        n = int(rng.integers(4, 30))
        diff = _draw(rng, n, rng.uniform(0.0, 0.4), 1.0, rng.normal() * 0.5)
        fit = _fit(diff, 0.1, mode, SHARP, 0.0)
        lo, hi = sharp_confint(diff, 0.95, fall_back_rho=0.1, mode=mode)
        np.testing.assert_allclose(lo, fit.mean - _Z95 * fit.se, rtol=1e-12)
        np.testing.assert_allclose(hi, fit.mean + _Z95 * fit.se, rtol=1e-12)


def test_interval_contains_the_point_estimate_and_is_ordered():
    rng = np.random.default_rng(10)
    for mode in VALID_MODES:
        for _ in range(20):
            n = int(rng.integers(4, 30))
            diff = _draw(rng, n, rng.uniform(0.0, 0.4), 1.0, rng.normal() * 0.5)
            lo, hi = sharp_confint(diff, 0.95, fall_back_rho=0.1, mode=mode)
            assert lo < float(np.mean(diff)) < hi


def test_interval_widens_with_the_level():
    rng = np.random.default_rng(11)
    for mode in VALID_MODES:
        diff = _draw(rng, 25, 0.2, 1.0, 0.5)
        widths = []
        for lv in (0.80, 0.90, 0.95, 0.99):
            lo, hi = sharp_confint(diff, lv, fall_back_rho=0.1, mode=mode)
            widths.append(hi - lo)
        assert all(a < b for a, b in zip(widths, widths[1:])), widths


def test_unbounded_when_the_level_is_out_of_reach():
    """``|z|`` is bounded, so a level the statistic cannot reach gives an
    interval that is unbounded rather than a silently wrong finite one."""
    rng = np.random.default_rng(12)
    diff = _draw(rng, 3, 0.2, 1.0, 5.0)
    # sqrt(2J) = 2.449 at J = 3, so a level needing |z| > 2.449 is unreachable.
    lo, hi = sharp_confint(diff, 0.999, mode="st")
    assert np.isneginf(lo) and np.isposinf(hi)
    # and a level well inside the ceiling is finite
    lo, hi = sharp_confint(diff, 0.50, mode="st")
    assert np.isfinite(lo) and np.isfinite(hi)


def test_coverage_is_close_to_nominal():
    """The point of the interval. Coverage is checked only where the score
    test is itself calibrated -- it is conservative at small ``rho``, and
    the interval inherits that, so a small-``rho`` cell would only pin the
    conservatism and would say nothing about the inversion."""
    rng = np.random.default_rng(13)
    draws = 400
    for n, mu in ((30, 0.0), (30, 0.5)):
        covered = sum(
            float(lo) <= mu <= float(hi)
            for lo, hi in (
                sharp_confint(_draw(rng, n, 0.45, 1.0, mu), 0.95)
                for _ in range(draws)
            )
        )
        assert 0.92 <= covered / draws <= 0.99, f"n={n} mu={mu}: {covered / draws}"


def test_bad_level_raises():
    diff = _draw(np.random.default_rng(14), 10, 0.2)
    for level in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="level must lie in"):
            sharp_confint(diff, level)


def test_invalid_mode_and_shape_raise():
    diff = _draw(np.random.default_rng(15), 10, 0.2)
    with pytest.raises(ValueError, match="mode must be"):
        sharp_confint(diff, 0.95, mode="bogus")
    with pytest.raises(ValueError, match=r"shape \[n, 2\]"):
        sharp_confint(np.zeros((5, 3)), 0.95)
    with pytest.raises(ValueError, match="fall_back_rho is required"):
        sharp_confint(diff, 0.95, mode="mm")


def test_degenerate_inputs_return_nan():
    lo, hi = sharp_confint(np.array([[0.1, 0.2]]), 0.95)
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = sharp_confint(np.full((5, 2), 0.05), 0.95)
    assert np.isnan(lo) and np.isnan(hi)


def test_sha_interval_is_reachable_through_the_engine():
    """SHA is switched off at its public entry point, but the dormant path
    must stay consistent with its own test."""
    from sharp_cv._engine import SHA

    rng = np.random.default_rng(16)
    cov = np.eye(20) + SHA.corr_pattern(10) * 0.3
    flat = 0.4 + np.linalg.cholesky(cov) @ rng.standard_normal(20)
    diff = flat.reshape(2, 10).T
    lo, hi = _split_half_confint(diff, 0.95, None, "st", SHA)
    p = _fit(diff, None, "st", SHA, 0.0).pvalue
    assert (lo <= 0.0 <= hi) == (p >= 0.05)
    # Bounded, but only because rho saturates and lets |z| past sqrt(2) on
    # the way out: the interval is ~95 times the spread of the data, which
    # is the same defect that switches SHA off.
    assert (hi - lo) > 50 * float(np.std(diff))
