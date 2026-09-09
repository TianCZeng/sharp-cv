"""sklearn-style entry point that runs the split-half procedure end to end.

:func:`sharp_cross_val_test` takes two sklearn estimators and ``(X, y)``,
runs the split-half procedure and hands the resulting paired differences
to :func:`~sharp_cv.sharp_test` or :func:`~sharp_cv.sha_test`. It also
accepts a pre-computed ``[n, 2]`` array through ``diff_AB``.

One *repetition* of the procedure is:

1. divide the data at random into two disjoint halves A and B (stratified
   by class for classifiers);
2. inside each half, run the inner cross-validation, fitting both
   estimators on the same training folds and scoring them on the same
   test folds;
3. reduce each half to one number, the mean over the inner folds of
   ``score_A - score_B``.

``cv`` sets both the inner cross-validation and the number of repetitions:

* ``RepeatedKFold(n_splits=K, n_repeats=J)`` or
  ``RepeatedStratifiedKFold(...)``: J repetitions, K-fold inside each half
  with the folds averaged. ``diff_AB`` is ``[J, 2]``; SHARP test.
* ``ShuffleSplit(n_splits=J, test_size=t)`` or
  ``StratifiedShuffleSplit(...)``: J repetitions, one train/test split
  inside each half. ``diff_AB`` is ``[J, 2]``; SHARP test.
* ``KFold(K)``, ``StratifiedKFold(K)`` or an int ``K``: a single
  repetition, K-fold inside each half with the folds kept as rows.
  ``diff_AB`` is ``[K, 2]``; SHA test.

Repeated and Monte-Carlo schemes redraw the halves on every repetition,
which is what the SHARP covariance model assumes. A single K-fold run
inside two fixed halves has a different covariance structure and is
handled by the SHA test.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
    check_cv,
    train_test_split,
)
from sklearn.utils import check_random_state

from sharp_cv._engine import VALID_MODES, sha_test, sharp_test

_VALID_TESTS = ("auto", "sharp", "sha")
_MAX_SEED = 2**31 - 1


class SharpTestResult(NamedTuple):
    """Result of :func:`sharp_cross_val_test`.

    Attributes:
        statistic: z statistic.
        pvalue: two-sided p-value.
        test: ``'sharp'`` or ``'sha'``, the test that produced the result.
        mode: estimator mode that was used.
        fall_back_rho: fallback correlation that was in effect.
        diff_AB: the ``[n, 2]`` paired-difference array that was tested.
    """

    statistic: float
    pvalue: float
    test: str
    mode: str
    fall_back_rho: float
    diff_AB: np.ndarray


def sharp_cross_val_test(
    estimator_a=None,
    estimator_b=None,
    X=None,
    y=None,
    *,
    diff_AB: Optional[np.ndarray] = None,
    cv=5,
    scoring=None,
    stratify: Optional[bool] = None,
    mode: str = "st",
    test: str = "auto",
    fall_back_rho: Optional[float] = None,
    random_state=None,
) -> SharpTestResult:
    """Compare two estimators with the SHARP or SHA split-half test.

    Args:
        estimator_a, estimator_b: sklearn-compatible estimators. Required
            unless ``diff_AB`` is given. Both are scored with the same
            scorer, so with ``scoring=None`` their ``score`` methods must
            measure the same thing.
        X, y: feature matrix and target. Required unless ``diff_AB`` is
            given.
        diff_AB: pre-computed ``[n, 2]`` array of paired split-half
            differences (model A minus model B). When given, the CV layer
            is skipped; ``test`` and ``fall_back_rho`` are then required
            because neither can be inferred from the array.
        cv: int or sklearn splitter selecting the inner cross-validation
            and the number of repetitions; see the module docstring. An
            int becomes ``StratifiedKFold`` for classifiers and ``KFold``
            otherwise.
        scoring: sklearn scoring string or callable. ``None`` uses the
            estimators' ``score`` method.
        stratify: ``None`` stratifies the half-split when ``estimator_a``
            is a classifier; ``True`` / ``False`` force the choice.
        mode: estimator mode, one of ``'mm'``, ``'mmc'``, ``'ml'``,
            ``'rml'``, ``'lrt'``, ``'st'`` (default). See
            :mod:`sharp_cv._engine`.
        test: with ``diff_AB``, ``'sharp'`` or ``'sha'`` stating how the
            rows were produced. With estimators it must stay ``'auto'``;
            the splitter decides which test is valid.
        fall_back_rho: correlation used by ``'mm'`` / ``'mmc'`` when the
            estimated variance falls below its minimum. With estimators
            the default is ``1 / (2 * K)`` for K-fold inner schemes and
            ``test_size / 2`` for Monte-Carlo schemes, where
            ``test_size`` is the fraction actually used inside each half.
            Required with ``diff_AB``.
        random_state: seed for the half-splits and, for repeated and
            Monte-Carlo schemes, for the inner splits as well. A plain
            ``KFold`` / ``StratifiedKFold`` object is used as given and
            keeps its own ``shuffle`` / ``random_state``.

    Returns:
        :class:`SharpTestResult`.
    """
    if mode not in VALID_MODES:
        hint = " mode='all' was removed; call once per mode." if mode == "all" else ""
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}.{hint}")
    if test not in _VALID_TESTS:
        raise ValueError(f"test must be one of {_VALID_TESTS}, got {test!r}")

    if diff_AB is not None:
        diff = np.asarray(diff_AB, dtype=float)
        if diff.ndim != 2 or diff.shape[1] != 2:
            raise ValueError(f"diff_AB must have shape [n, 2]; got {diff.shape}")
        if test == "auto":
            raise ValueError(
                "test cannot be inferred from diff_AB. Pass test='sharp' if "
                "each row is one repetition of the split-half procedure "
                "(halves redrawn every time), or test='sha' if the rows are "
                "the folds of a single K-fold run inside two fixed halves."
            )
        if fall_back_rho is None:
            raise ValueError(
                "fall_back_rho is required with diff_AB. Use 1 / (2 * K) when "
                "a K-fold CV was run inside each half, or test_size / 2 for a "
                "single Monte-Carlo split inside each half."
            )
        which = test
    else:
        if estimator_a is None or estimator_b is None or X is None or y is None:
            raise ValueError(
                "Provide either diff_AB=..., or all of (estimator_a, "
                "estimator_b, X, y)."
            )
        if test != "auto":
            raise ValueError(
                "test must be 'auto' when estimators are given: the splitter "
                "decides which test is valid (SHARP for repeated K-fold and "
                "Monte-Carlo schemes, SHA for a single K-fold). Use test= "
                "only together with diff_AB."
            )
        diff, which, default_rho = _build_diff_AB(
            estimator_a, estimator_b, X, y,
            cv=cv, scoring=scoring, stratify=stratify, random_state=random_state,
        )
        if fall_back_rho is None:
            fall_back_rho = default_rho

    run = sha_test if which == "sha" else sharp_test
    z, p = run(diff, fall_back_rho=fall_back_rho, mode=mode)
    return SharpTestResult(
        statistic=float(z), pvalue=float(p), test=which, mode=mode,
        fall_back_rho=float(fall_back_rho), diff_AB=diff,
    )


# ---------------------------------------------------------------------------
# Procedure
# ---------------------------------------------------------------------------

class _Scheme(NamedTuple):
    """How ``cv`` maps onto the split-half procedure.

    ``kind`` is ``'repeated_kfold'``, ``'monte_carlo'`` or ``'kfold'``.
    ``n_repeats`` is the number of independent half-splits.
    ``make_inner(rng)`` returns the splitter run inside each half of one
    repetition.
    """

    kind: str
    n_repeats: int
    make_inner: Callable[[np.random.RandomState], Any]


def _resolve_scheme(cv, y, is_clf: bool) -> _Scheme:
    splitter = check_cv(cv, y, classifier=is_clf)

    # RepeatedKFold / RepeatedStratifiedKFold and user subclasses of
    # sklearn's _RepeatedSplits: one fresh shuffled K-fold per repetition.
    if all(hasattr(splitter, a) for a in ("n_repeats", "cv", "cvargs")):
        base, kwargs = splitter.cv, dict(splitter.cvargs)
        n_repeats = int(splitter.n_repeats)

        def make_inner(rng):
            return base(shuffle=True, random_state=rng.randint(_MAX_SEED), **kwargs)

        return _Scheme("repeated_kfold", n_repeats, make_inner)

    # ShuffleSplit / StratifiedShuffleSplit: one train/test split per half
    # per repetition, with the user's test_size / train_size.
    if isinstance(splitter, (ShuffleSplit, StratifiedShuffleSplit)):
        cls, n_repeats = type(splitter), int(splitter.n_splits)
        test_size, train_size = splitter.test_size, splitter.train_size

        def make_inner(rng):
            return cls(
                n_splits=1, test_size=test_size, train_size=train_size,
                random_state=rng.randint(_MAX_SEED),
            )

        return _Scheme("monte_carlo", n_repeats, make_inner)

    # Anything else is run once inside each half of a single half-split.
    return _Scheme("kfold", 1, lambda rng: splitter)


def _build_diff_AB(estimator_a, estimator_b, X, y, *, cv=5, scoring=None,
                   stratify: Optional[bool] = None, random_state=None):
    """Run the split-half procedure.

    Returns ``(diff_AB, test, default_fall_back_rho)`` where ``test`` is
    ``'sharp'`` or ``'sha'``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of rows; got {X.shape[0]} and {y.shape[0]}"
        )

    is_clf = stratify if isinstance(stratify, bool) else is_classifier(estimator_a)
    rng = check_random_state(random_state)
    scheme = _resolve_scheme(cv, y, is_clf)
    if scheme.kind != "kfold" and scheme.n_repeats < 2:
        raise ValueError(
            "SHARP needs at least 2 repetitions; got "
            f"{scheme.n_repeats} (n_repeats / n_splits of the splitter)."
        )
    scorer = check_scoring(estimator_a, scoring=scoring)

    repetitions = []   # one [diffs_half_A, diffs_half_B] per repetition
    test_fractions = []
    for _ in range(scheme.n_repeats):
        X_A, X_B, y_A, y_B = train_test_split(
            X, y, test_size=0.5,
            stratify=y if is_clf else None,
            random_state=rng.randint(_MAX_SEED),
        )
        inner = scheme.make_inner(rng)
        halves = []
        for X_h, y_h in ((X_A, y_A), (X_B, y_B)):
            diffs = []
            for train_idx, test_idx in inner.split(X_h, y_h):
                a = clone(estimator_a).fit(X_h[train_idx], y_h[train_idx])
                b = clone(estimator_b).fit(X_h[train_idx], y_h[train_idx])
                diffs.append(
                    scorer(a, X_h[test_idx], y_h[test_idx])
                    - scorer(b, X_h[test_idx], y_h[test_idx])
                )
                test_fractions.append(len(test_idx) / len(y_h))
            halves.append(diffs)
        if len(halves[0]) != len(halves[1]):
            raise RuntimeError(
                "The two halves produced different numbers of inner splits "
                f"({len(halves[0])} vs {len(halves[1])}); use a splitter with "
                "a fixed n_splits."
            )
        repetitions.append(halves)

    n_inner = len(repetitions[0][0])

    if scheme.kind == "kfold":
        # Single repetition: keep the per-fold rows.
        diff = np.column_stack(repetitions[0])
        return diff, "sha", 1.0 / (2 * n_inner)

    # Repeated schemes: one row per repetition, folds averaged within a half.
    diff = np.array([[np.mean(half_A), np.mean(half_B)] for half_A, half_B in repetitions])
    if scheme.kind == "monte_carlo":
        return diff, "sharp", float(np.mean(test_fractions)) / 2.0
    return diff, "sharp", 1.0 / (2 * n_inner)
