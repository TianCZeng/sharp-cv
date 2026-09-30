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
from sharp_cv._engine import sha_test

REF = json.loads(
    (Path(__file__).parent / "reference_data" / "sharp_reference.json").read_text()
)
CASES = REF["cases"]
IDS = [c["name"] for c in CASES]

# The method-of-moments estimators are exact and are compared tightly. The
# reference values for the likelihood-based modes were recorded from an
# earlier search that stopped at a BFGS gradient tolerance of 1e-4, so they
# are not pinned any tighter than that; the current fits reproduce them to
# within 4e-5 relative in z. 1e-4 keeps z to four significant figures, which
# is far finer than any real change to an estimator.
_TOL = {"mm": 1e-10, "mmc": 1e-10, "ml": 1e-4, "rml": 1e-4, "lrt": 1e-4, "st": 1e-4}


def _p_tol(mode, z_ref):
    """Tolerance on p implied by the tolerance on z.

    p is a two-sided normal tail probability, so a relative error eps in z
    leaves it with a relative error of about z**2 * eps -- far out in the
    tail the same eps is worth much more in p than in z.
    """
    return _TOL[mode] * max(1.0, float(z_ref) ** 2)


@pytest.mark.parametrize("mode", VALID_MODES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_matches_cbig_reference(case, mode):
    diff = np.asarray(case["diff_AB"])
    z_ref, p_ref = case["expected"][mode]
    z, p = sharp_test(diff, fall_back_rho=case["fall_back_rho"], mode=mode)
    np.testing.assert_allclose(z, z_ref, rtol=_TOL[mode], atol=1e-12)
    np.testing.assert_allclose(p, p_ref, rtol=_p_tol(mode, z_ref), atol=1e-12)


def test_degenerate_and_non_finite_inputs_return_nan():
    """One row, identical halves, or a NaN or inf anywhere in the input."""
    inputs = [np.array([[0.1, 0.2]]), np.full((5, 2), 0.05)]
    for bad in (np.nan, np.inf):
        diff = np.array(CASES[0]["diff_AB"], dtype=float)
        diff[0, 0] = bad
        inputs.append(diff)
    for mode in VALID_MODES:
        for diff in inputs:
            with np.errstate(invalid="ignore"):
                z, p = sharp_test(diff, fall_back_rho=0.1, mode=mode)
            assert np.isnan(z) and np.isnan(p), mode


def test_invalid_arguments_raise():
    diff = np.asarray(CASES[0]["diff_AB"])
    with pytest.raises(ValueError, match="mode must be"):
        sharp_test(diff, fall_back_rho=0.1, mode="bogus")
    with pytest.raises(ValueError, match="mode='all' is not supported"):
        sharp_test(diff, fall_back_rho=0.1, mode="all")
    for bad_shape in (np.zeros((5, 3)), np.zeros(10)):
        with pytest.raises(ValueError, match=r"shape \[n, 2\]"):
            sharp_test(bad_shape, fall_back_rho=0.1)
    with pytest.raises(ValueError, match="fall_back_rho must be a finite number"):
        sharp_test(diff, fall_back_rho=float("nan"))
    for mode in ("mm", "mmc"):
        with pytest.raises(ValueError, match="fall_back_rho is required"):
            sharp_test(diff, mode=mode)


def test_sha_test_is_switched_off():
    with pytest.raises(NotImplementedError, match="SHA test is not available"):
        sha_test(np.asarray(CASES[0]["diff_AB"]), fall_back_rho=0.1)
