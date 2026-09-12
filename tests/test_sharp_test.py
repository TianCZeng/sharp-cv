"""Tests for sharp_cv.sharp_test.

The expected values in ``reference_data/sharp_reference.json`` were produced
by the implementation used for the manuscript, except ``'rml'``, which comes
from sharp_cv (see the ``_provenance`` entry in the file). Inputs are stored
verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sharp_cv import VALID_MODES, sharp_test
from sharp_cv._engine import SHARP

REF = json.loads(
    (Path(__file__).parent / "reference_data" / "sharp_reference.json").read_text()
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


def _mom_z(diff, fall_back_rho, var_of_mean):
    """Method-of-moments z with the fallback rule, written out by hand."""
    n = diff.shape[0]
    A, B = diff[:, 0], diff[:, 1]
    sig2 = np.mean((A - B) ** 2) / 2
    rho = 1 - 0.5 * (np.var(A, ddof=1) + np.var(B, ddof=1)) / sig2
    var = var_of_mean(n, sig2, rho)
    if var < np.var(diff, ddof=1) / (2 * n):
        var = var_of_mean(n, sig2, fall_back_rho)
    return diff.mean() / np.sqrt(var)


@pytest.mark.parametrize("mode", VALID_MODES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_matches_cbig_reference(case, mode):
    diff = np.asarray(case["diff_AB"])
    z_ref, p_ref = case["expected"][mode]
    z, p = sharp_test(diff, fall_back_rho=case["fall_back_rho"], mode=mode)
    np.testing.assert_allclose(z, z_ref, rtol=_TOL[mode], atol=1e-12)
    np.testing.assert_allclose(p, p_ref, rtol=_p_tol(mode, z_ref), atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mm_matches_closed_form(case):
    diff = np.asarray(case["diff_AB"])
    z, _ = sharp_test(diff, fall_back_rho=case["fall_back_rho"], mode="mm")
    z_ref = _mom_z(diff, case["fall_back_rho"], SHARP.var_of_mean)
    np.testing.assert_allclose(z, z_ref, rtol=1e-12)


def test_reference_covers_both_fallback_branches():
    flags = {c["fallback_triggered_mm"] for c in CASES}
    assert flags == {True, False}


def test_accepts_list_input():
    diff = [[0.1, 0.12], [0.08, 0.09], [0.11, 0.13], [0.07, 0.10], [0.09, 0.11]]
    z, p = sharp_test(diff, fall_back_rho=0.1, mode="mm")
    assert np.isfinite(z) and 0.0 <= p <= 1.0


def test_degenerate_inputs_return_nan():
    z, p = sharp_test(np.array([[0.1, 0.2]]), fall_back_rho=0.1, mode="mm")
    assert np.isnan(z) and np.isnan(p)
    diff = np.full((5, 2), 0.05)  # identical halves
    z, p = sharp_test(diff, fall_back_rho=0.1, mode="st")
    assert np.isnan(z) and np.isnan(p)


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        sharp_test(np.asarray(CASES[0]["diff_AB"]), fall_back_rho=0.1, mode="bogus")


def test_mode_all_is_rejected():
    with pytest.raises(ValueError, match="mode='all' is not supported"):
        sharp_test(np.asarray(CASES[0]["diff_AB"]), fall_back_rho=0.1, mode="all")


def test_bad_shape_raises():
    with pytest.raises(ValueError, match=r"shape \[n, 2\]"):
        sharp_test(np.zeros((5, 3)), fall_back_rho=0.1)
    with pytest.raises(ValueError, match=r"shape \[n, 2\]"):
        sharp_test(np.zeros(10), fall_back_rho=0.1)


def test_non_finite_fall_back_rho_raises():
    with pytest.raises(ValueError, match="fall_back_rho"):
        sharp_test(np.asarray(CASES[0]["diff_AB"]), fall_back_rho=float("nan"))


def test_fall_back_rho_required_only_for_mm_and_mmc():
    diff = np.asarray(CASES[0]["diff_AB"])
    for mode in ("mm", "mmc"):
        with pytest.raises(ValueError, match="fall_back_rho is required"):
            sharp_test(diff, mode=mode)
    for mode in ("ml", "rml", "lrt", "st"):
        assert sharp_test(diff, mode=mode) == sharp_test(diff, 0.3, mode=mode)


@pytest.mark.parametrize("mode", VALID_MODES)
def test_pvalue_in_unit_interval(mode):
    for case in CASES:
        _, p = sharp_test(np.asarray(case["diff_AB"]), fall_back_rho=0.1, mode=mode)
        assert 0.0 <= p <= 1.0
