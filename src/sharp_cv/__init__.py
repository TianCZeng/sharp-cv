"""SHARP split-half cross-validation test.

A statistical test for comparing the predictive performance of two models
under a re-designed cross-validation scheme.

:func:`sharp_cross_val_test` runs the procedure on two sklearn estimators
and tests the result. :func:`sharp_test` tests paired split-half
differences that you computed yourself, and :func:`sharp_confint` puts an
interval around the difference it estimates.
"""

from sharp_cv._engine import VALID_MODES, sharp_confint, sharp_test
from sharp_cv._version import __version__
from sharp_cv.cross_val import SharpTestResult, sharp_cross_val_test

__all__ = [
    "sharp_test",
    "sharp_confint",
    "sharp_cross_val_test",
    "SharpTestResult",
    "VALID_MODES",
    "__version__",
]
