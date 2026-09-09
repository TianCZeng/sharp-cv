"""Tests for sharp_cv.sha_test.

SHA has no independent reference implementation. The expected values in
``reference_data/sha_reference.json`` were recorded from sharp_cv 0.1.0
before the engine refactor and guard against regressions; the closed-form
checks below verify the method-of-moments path independently.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sharp_cv import VALID_MODES, sha_test
from sharp_cv._engine import SHA, SHARP, _split_half_test

REF = json.loads(
    (Path(__file__).parent / "reference_data" / "sha_reference.json").read_text()
)
CASES = REF["cases"]
IDS = [c["name"] for c in CASES]
# Closed-form estimators are compared tightly. Likelihood-based modes stop
# at a BFGS gradient tolerance of 1e-4, and SciPy versions differ in the
# last optimiser steps by up to ~4e-7 relative in z, so they get 1e-5.
_TOL = {"mm": 1e-10, "mmc": 1e-10, "ml": 1e-5, "rml": 1e-5, "lrt": 1e-5, "st": 1e-5}


def _mom_z(diff, fall_back_rho, var_of_mean):
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
def test_matches_recorded_reference(case, mode):
    diff = np.asarray(case["diff_AB"])
    z_ref, p_ref = case["expected"][mode]
    z, p = sha_test(diff, fall_back_rho=case["fall_back_rho"], mode=mode)
    np.testing.assert_allclose(z, z_ref, rtol=_TOL[mode], atol=1e-12)
    np.testing.assert_allclose(p, p_ref, rtol=_TOL[mode], atol=1e-12)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mm_matches_closed_form(case):
    """Var(mean) under SHA is sigma^2 (1 + (K-1) rho) / (2K), with the
    fallback rule applied when it drops below the naive minimum."""
    diff = np.asarray(case["diff_AB"])
    z, _ = sha_test(diff, fall_back_rho=case["fall_back_rho"], mode="mm")
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
        assert _split_half_test(diff, fb, "mm", SHARP).se >= _split_half_test(diff, fb, "mm", SHA).se
        checked += 1
    assert checked > 0


def test_degenerate_inputs_return_nan():
    z, p = sha_test(np.array([[0.1, 0.2]]), fall_back_rho=0.1, mode="mm")
    assert np.isnan(z) and np.isnan(p)
    z, p = sha_test(np.full((5, 2), 0.05), fall_back_rho=0.1, mode="st")
    assert np.isnan(z) and np.isnan(p)


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        sha_test(np.asarray(CASES[0]["diff_AB"]), fall_back_rho=0.1, mode="bogus")


def test_mode_all_is_rejected():
    with pytest.raises(ValueError, match="mode='all' is not supported"):
        sha_test(np.asarray(CASES[0]["diff_AB"]), fall_back_rho=0.1, mode="all")


@pytest.mark.parametrize("mode", VALID_MODES)
def test_pvalue_in_unit_interval(mode):
    for case in CASES:
        _, p = sha_test(np.asarray(case["diff_AB"]), fall_back_rho=0.1, mode=mode)
        assert 0.0 <= p <= 1.0
