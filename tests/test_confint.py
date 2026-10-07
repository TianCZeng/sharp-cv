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
from sharp_cv._engine import SHARP, _fit

_Z95 = float(stats.norm.isf(0.025))


def _draw(rng, n, rho, sigma=1.0, mu=0.0):
    cov = (np.eye(2 * n) + SHARP.corr_pattern(n) * rho) * sigma**2
    flat = mu + np.linalg.cholesky(cov) @ rng.standard_normal(2 * n)
    return flat.reshape(2, n).T


@pytest.mark.parametrize("mode", VALID_MODES)
def test_interval_agrees_with_the_pvalue(mode):
    """0 lies inside the interval exactly when the test does not reject.

    This is the property the interval exists to have, and the reason
    ``'st'`` and ``'lrt'`` re-estimate the nuisance parameters at every
    hypothesised mean instead of reusing the null fit.
    """
    rng = np.random.default_rng(7)
    for _ in range(120):
        n = int(rng.integers(4, 40))
        diff = _draw(
            rng, n, rng.uniform(0.0, 0.45), rng.uniform(0.2, 2.0), rng.normal() * 0.4
        )
        p = sharp_test(diff, fall_back_rho=0.1, mode=mode)[1]
        lo, hi = sharp_confint(diff, 0.95, fall_back_rho=0.1, mode=mode)
        assert (lo <= 0.0 <= hi) == (p >= 0.05), f"p={p} interval=({lo}, {hi})"


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
