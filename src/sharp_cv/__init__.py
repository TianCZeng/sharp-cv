"""SHARP split-half cross-validation test.

A statistical test for comparing the predictive performance of two models
under a re-designed cross-validation scheme.

The single-run SHA variant is switched off in this release: :func:`sha_test`
is still importable but raises, with the reason in the message.
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
