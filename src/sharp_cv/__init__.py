"""SHARP and SHA split-half cross-validation tests.

Statistical tests for comparing the predictive performance of two models
under a re-designed cross-validation scheme.
"""

from sharp_cv._engine import VALID_MODES, sha_test, sharp_test
from sharp_cv._version import __version__
from sharp_cv.cross_val import SharpTestResult, sharp_cross_val_test

__all__ = [
    "sharp_test",
    "sha_test",
    "sharp_cross_val_test",
    "SharpTestResult",
    "VALID_MODES",
    "__version__",
]
