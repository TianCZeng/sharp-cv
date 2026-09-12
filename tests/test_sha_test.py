"""Tests for the SHA correlation pattern.

SHA is switched off: ``sha_test`` raises and is not exported from
:mod:`sharp_cv`, so these tests reach the estimators through
``_split_half_test(..., SHA)`` instead. They keep the dormant path covered
for when a working estimator replaces the current one.

SHA has no independent reference implementation. The expected values in
``reference_data/sha_reference.json`` were recorded from sharp_cv and guard
against regressions; the closed-form checks below verify the
method-of-moments path independently,
``test_rho_bound_follows_the_covariance_pattern`` checks that SHA's bound on
rho follows its own covariance rather than SHARP's, and
``test_score_statistic_is_bounded_and_cannot_reject`` pins the defect that
the switch is there for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sharp_cv import VALID_MODES
from sharp_cv._engine import SHA, SHARP, Structure, _split_half_test, sha_test

REF = json.loads(
    (Path(__file__).parent / "reference_data" / "sha_reference.json").read_text()
)
CASES = REF["cases"]
IDS = [c["name"] for c in CASES]
# The closed-form estimators are exact and are compared tightly. The
# likelihood-based modes stop at a BFGS gradient tolerance of 1e-4, so where
# the search stops is not pinned any tighter than that: across SciPy versions
# and BLAS thread counts z moves by up to 2e-6 relative, and a GitHub runner
# was seen 3x beyond that on the worst case ('rml', model_seed10_J8_rho0.3).
# 1e-4 keeps z to four significant figures, which is far finer than any real
# change to an estimator, and leaves room for the next machine.
_TOL = {"mm": 1e-10, "mmc": 1e-10, "ml": 1e-4, "rml": 1e-4, "lrt": 1e-4, "st": 1e-4}


def _p_tol(mode, z_ref):
    """Tolerance on p implied by the tolerance on z.

    p is a two-sided normal tail probability, so a relative error eps in z
    leaves it with a relative error of about z**2 * eps -- far out in the
    tail the same eps is worth much more in p than in z.
    """
    return _TOL[mode] * max(1.0, float(z_ref) ** 2)


def _sha(diff_AB, fall_back_rho=None, mode="st"):
    """What ``sha_test`` would return if it were switched on."""
    fit = _split_half_test(diff_AB, fall_back_rho, mode, SHA)
    return fit.statistic, fit.pvalue


def _mom_z(diff, fall_back_rho, var_of_mean):
    n = diff.shape[0]
    A, B = diff[:, 0], diff[:, 1]
    sig2 = np.mean((A - B) ** 2) / 2
    rho = 1 - 0.5 * (np.var(A, ddof=1) + np.var(B, ddof=1)) / sig2
    var = var_of_mean(n, sig2, rho)
    if var < np.var(diff, ddof=1) / (2 * n):
        var = var_of_mean(n, sig2, fall_back_rho)
    return diff.mean() / np.sqrt(var)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", VALID_MODES)
def test_sha_test_is_switched_off(mode):
    diff = np.asarray(CASES[0]["diff_AB"])
    with pytest.raises(NotImplementedError, match="SHA test is not available"):
        sha_test(diff, fall_back_rho=0.1, mode=mode)


def test_switch_fires_before_input_validation():
    """The message must explain SHA rather than complain about the input."""
    with pytest.raises(NotImplementedError, match="SHA test is not available"):
        sha_test(np.zeros((5, 3)), mode="bogus")


def test_sha_test_is_not_exported():
    import sharp_cv

    assert "sha_test" not in sharp_cv.__all__
    assert not hasattr(sharp_cv, "sha_test")


def test_score_statistic_is_bounded_and_cannot_reject():
    """Why SHA is switched off.

    The variance of the SHA mean is fixed by the two half means alone, and
    the score test estimates it under the null from those same two numbers.
    Statistic and standard error then rise together and cancel: |z| is
    pinned at sqrt(2), so p never falls below 0.157 however large the true
    difference is, and power at any conventional threshold is zero.
    """
    K, rho = 10, 0.3
    cov = np.eye(2 * K) + SHA.corr_pattern(K) * rho
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(5)
    for mu in (0.0, 1.0, 5.0):
        draws = chol @ rng.standard_normal((2 * K, 300)) + mu
        zp = [
            _sha(np.column_stack([draws[:K, j], draws[K:, j]]), mode="st")
            for j in range(draws.shape[1])
        ]
        z = np.abs([z for z, _ in zp])
        p = np.array([p for _, p in zp])
        assert z.max() <= np.sqrt(2) * (1 + 1e-3)
        assert p.min() >= 0.156
        assert (p < 0.05).sum() == 0


# ---------------------------------------------------------------------------
# The dormant estimators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", VALID_MODES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_matches_recorded_reference(case, mode):
    diff = np.asarray(case["diff_AB"])
    z_ref, p_ref = case["expected"][mode]
    z, p = _sha(diff, fall_back_rho=case["fall_back_rho"], mode=mode)
    np.testing.assert_allclose(z, z_ref, rtol=_TOL[mode], atol=1e-12)
    np.testing.assert_allclose(p, p_ref, rtol=_p_tol(mode, z_ref), atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mm_matches_closed_form(case):
    """Var(mean) under SHA is sigma^2 (1 + (K-1) rho) / (2K), with the
    fallback rule applied when it drops below the naive minimum."""
    diff = np.asarray(case["diff_AB"])
    z, _ = _sha(diff, fall_back_rho=case["fall_back_rho"], mode="mm")
    z_ref = _mom_z(diff, case["fall_back_rho"], SHA.var_of_mean)
    np.testing.assert_allclose(z, z_ref, rtol=1e-12)


def test_reference_covers_both_fallback_branches():
    flags = {c["fallback_triggered_mm"] for c in CASES}
    assert flags == {True, False}


def test_sha_variance_never_exceeds_sharp_at_nonnegative_rho():
    """For the same data, SHA loads rho with half the weight SHARP does, so
    its MoM standard error is at most SHARP's whenever rho_hat >= 0."""
    checked = 0
    for case in CASES:
        diff = np.asarray(case["diff_AB"])
        A, B = diff[:, 0], diff[:, 1]
        sig2 = np.mean((A - B) ** 2) / 2
        rho = 1 - 0.5 * (np.var(A, ddof=1) + np.var(B, ddof=1)) / sig2
        if rho < 0:
            continue
        fb = case["fall_back_rho"]
        se_sharp = _split_half_test(diff, fb, "mm", SHARP).se
        se_sha = _split_half_test(diff, fb, "mm", SHA).se
        assert se_sharp >= se_sha
        checked += 1
    assert checked > 0


def test_rho_bound_follows_the_covariance_pattern():
    """SHA's covariance is block diagonal, with eigenvalues sigma^2 (1 - rho)
    and sigma^2 (1 + (K-1) rho), so it stays positive definite up to rho = 1.
    SHARP's is singular at rho = 0.5. The bound on rho must follow the
    pattern, or SHA saturates and understates the variance of the mean."""
    assert SHARP.rho_max < 0.5 <= SHA.rho_max < 1.0
    for structure in (SHARP, SHA):
        assert 0.0 < structure.rho_clip < structure.rho_max
        pattern = structure.corr_pattern(6)
        eye = np.eye(pattern.shape[0])
        at_max = np.linalg.eigvalsh(eye + pattern * structure.rho_max)
        assert at_max.min() > 0, f"{structure.name} not PD at its own rho_max"
        beyond = np.linalg.eigvalsh(eye + pattern * min(structure.rho_max + 0.01, 1.0))
        assert beyond.min() < at_max.min()


def test_capping_sha_at_the_sharp_bound_would_inflate_z():
    """'seed5_K5' has a within-half correlation above 0.5. Running SHA with
    SHARP's bound on rho saturates rho there and shrinks the standard error,
    so |z| must be larger under the SHARP-capped bound than under SHA's own."""
    case = next(c for c in CASES if c["name"] == "seed5_K5")
    diff = np.asarray(case["diff_AB"])
    capped = Structure("sha_capped", SHA.corr_pattern, SHA.var_of_mean, SHARP.rho_max)
    for mode in ("st", "lrt"):
        z_sha = _split_half_test(diff, case["fall_back_rho"], mode, SHA).statistic
        z_capped = _split_half_test(diff, case["fall_back_rho"], mode, capped).statistic
        assert abs(z_sha) < abs(z_capped)


def test_fall_back_rho_required_only_for_mm_and_mmc():
    diff = np.asarray(CASES[0]["diff_AB"])
    for mode in ("mm", "mmc"):
        with pytest.raises(ValueError, match="fall_back_rho is required"):
            _sha(diff, mode=mode)
    for mode in ("ml", "rml", "lrt", "st"):
        assert _sha(diff, mode=mode) == _sha(diff, 0.3, mode=mode)


def test_degenerate_inputs_return_nan():
    z, p = _sha(np.array([[0.1, 0.2]]), fall_back_rho=0.1, mode="mm")
    assert np.isnan(z) and np.isnan(p)
    z, p = _sha(np.full((5, 2), 0.05), fall_back_rho=0.1, mode="st")
    assert np.isnan(z) and np.isnan(p)


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        _sha(np.asarray(CASES[0]["diff_AB"]), fall_back_rho=0.1, mode="bogus")


def test_mode_all_is_rejected():
    with pytest.raises(ValueError, match="mode='all' is not supported"):
        _sha(np.asarray(CASES[0]["diff_AB"]), fall_back_rho=0.1, mode="all")


@pytest.mark.parametrize("mode", VALID_MODES)
def test_pvalue_in_unit_interval(mode):
    for case in CASES:
        _, p = _sha(np.asarray(case["diff_AB"]), fall_back_rho=0.1, mode=mode)
        assert 0.0 <= p <= 1.0
