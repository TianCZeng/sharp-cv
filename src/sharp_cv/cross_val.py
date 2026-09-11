"""sklearn-style entry point that runs the split-half procedure end to end.

:func:`sharp_cross_val_test` takes two sklearn estimators and ``(X, y)``,
runs the split-half procedure and hands the resulting paired differences
to :func:`~sharp_cv.sharp_test` or :func:`~sharp_cv.sha_test`. It also
accepts a pre-computed ``[n, 2]`` array through ``diff_AB``, whose ``AB``
refers to the two halves of the data, not to the two models.

One *repetition* of the procedure is:

1. divide the data at random into two disjoint halves A and B (stratified
   by class for classifiers);
2. inside each half, run the inner cross-validation, fitting both
   estimators on the same training folds and scoring them on the same
   test folds;
3. reduce each half to one mean difference, the mean over
   ``score_1 - score_2``.

``cv`` is required, and sets both the inner cross-validation and the
number of repetitions:

* ``RepeatedKFold(n_splits=K, n_repeats=J)`` or
  ``RepeatedStratifiedKFold(...)``: J repetitions, K-fold inside each half
  with the folds averaged. ``diff_AB`` is ``[J, 2]``; SHARP test will be
  used. Around ``J = 30`` is a reasonable starting point.
* ``ShuffleSplit(n_splits=J, test_size=t)`` or
  ``StratifiedShuffleSplit(...)``: J repetitions, one train/test split
  inside each half. ``diff_AB`` is ``[J, 2]``; SHARP test will be used.
  ``n_splits`` counts repetitions here, not inner folds; around
  ``J = 150`` is a reasonable starting point.
* ``KFold(K)``, ``StratifiedKFold(K)`` or an int ``K``: a single
  repetition, which only the SHA test can analyse. Rejected while SHA is
  switched off; see :mod:`sharp_cv._engine`.

For the repeated and Monte-Carlo splitters only ``n_splits``,
``n_repeats``, ``test_size`` and ``train_size`` are read; their ``split``
method is never called and their ``random_state`` is ignored (with a
warning when one is set), because every split is drawn from the
``random_state`` given to :func:`sharp_cross_val_test`. A plain ``KFold``
/ ``StratifiedKFold`` is used as given. Group-aware splitters such as
``GroupKFold`` are rejected: there is no ``groups`` argument, and the
half-split would place samples of one group in both halves.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, NamedTuple

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
    check_cv,
    train_test_split,
)
from sklearn.utils import check_random_state, indexable

from sharp_cv._engine import VALID_MODES, _require_sha, sha_test, sharp_test

_VALID_TESTS = ("auto", "sharp", "sha")
_MAX_SEED = 2**31 - 1
# Rejected by _resolve_scheme: there is no groups argument to honour.
_GROUP_SPLITTERS = (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedGroupKFold,
)


class SharpTestResult(NamedTuple):
    """Result of :func:`sharp_cross_val_test`.

    Attributes:
        statistic: z statistic.
        pvalue: two-sided p-value.
        test: the test that produced the result, always ``'sharp'`` while
            SHA is switched off.
        mode: estimator mode that was used.
        fall_back_rho: fallback correlation that was in effect, or ``None``
            when ``diff_AB`` was given without one and the mode does not
            use it.
        diff_AB: the ``[n, 2]`` paired-difference array that was tested.
    """

    statistic: float
    pvalue: float
    test: str
    mode: str
    fall_back_rho: float | None
    diff_AB: np.ndarray


def sharp_cross_val_test(
    estimator_1=None,
    estimator_2=None,
    X=None,
    y=None,
    *,
    diff_AB: np.ndarray | None = None,
    cv=None,
    scoring=None,
    stratify: bool | None = None,
    mode: str = "st",
    test: str = "auto",
    fall_back_rho: float | None = None,
    random_state=None,
) -> SharpTestResult:
    """Compare two estimators with the SHARP or SHA split-half test.

    Args:
        estimator_1, estimator_2: sklearn-compatible estimators. Required
            unless ``diff_AB`` is given. Both are scored with the same
            scorer, so with ``scoring=None`` their ``score`` methods must
            measure the same thing. The tested difference is
            ``estimator_1`` minus ``estimator_2``. A ``GridSearchCV``, or a
            pipeline containing one, is refit inside every training fold,
            which gives nested cross-validation.
        X, y: feature matrix and target. Required unless ``diff_AB`` is
            given. NumPy arrays, pandas objects and SciPy sparse matrices
            are accepted; rows are selected with ``.iloc`` for pandas, so
            column names reach the estimators.
        diff_AB: pre-computed ``[n, 2]`` array of paired split-half
            differences (model 1 minus model 2), with column 0 from half A
            and column 1 from half B. When given, the CV layer is skipped
            and ``test`` is required, because it cannot be inferred from
            the array.
        cv: sklearn splitter selecting the inner cross-validation and the
            number of split-half repetitions; see the module docstring.
            Required when estimators are given, ignored with ``diff_AB``.
            It has no default because the repetition count is a choice:
            around 30 for ``RepeatedKFold`` / ``RepeatedStratifiedKFold``,
            around 150 for ``ShuffleSplit`` / ``StratifiedShuffleSplit``.
            Single-run splitters (an int, ``KFold``, ``StratifiedKFold``)
            need the SHA test and are rejected, as are group-aware
            splitters.
        scoring: sklearn scoring string or callable. ``None`` uses the
            estimators' ``score`` method.
        stratify: ``None`` stratifies the half-split when ``estimator_1``
            is a classifier; ``True`` / ``False`` force the choice.
        mode: estimator mode, one of ``'mm'``, ``'mmc'``, ``'ml'``,
            ``'rml'``, ``'lrt'``, ``'st'`` (default). See
            :mod:`sharp_cv._engine`.
        test: with ``diff_AB``, ``'sharp'`` stating that each row is one
            repetition of the split-half procedure; it cannot be inferred
            from the array. ``'sha'`` is rejected while SHA is switched
            off. With estimators it must stay ``'auto'``; the splitter
            decides which test is valid.
        fall_back_rho: correlation used by ``'mm'`` / ``'mmc'`` when the
            estimated variance falls below its minimum; ignored by every
            other mode, including the default. With estimators the default
            is ``1 / (2 * K)`` for K-fold inner schemes and
            ``test_size / 2`` for Monte-Carlo schemes, where ``test_size``
            is the fraction actually used inside each half. Halving is what
            expresses these as a fraction of the full dataset, since an
            inner test set is drawn from a half. With ``diff_AB`` it is
            required for ``'mm'`` and ``'mmc'`` and may be left as ``None``
            otherwise.
        random_state: seed for the half-splits and, for repeated and
            Monte-Carlo schemes, for the inner splits as well. The
            ``random_state`` of a ``RepeatedKFold``,
            ``RepeatedStratifiedKFold``, ``ShuffleSplit`` or
            ``StratifiedShuffleSplit`` passed as ``cv`` is ignored, with a
            warning when one is set. A plain ``KFold`` / ``StratifiedKFold``
            object is used as given and keeps its own ``shuffle`` /
            ``random_state``; when it shuffles, give it a ``random_state``
            too, or the result is not reproducible (a warning is raised in
            that case).

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
                "test cannot be inferred from diff_AB. Pass test='sharp': "
                "each row must be one repetition of the split-half "
                "procedure, with the halves redrawn every time."
            )
        if test == "sha":
            _require_sha()
        which = test
    else:
        if estimator_1 is None or estimator_2 is None or X is None or y is None:
            raise ValueError(
                "Provide either diff_AB=..., or all of (estimator_1, "
                "estimator_2, X, y)."
            )
        if cv is None:
            raise ValueError(
                "cv is required. Pass a repeated splitter, for example "
                "RepeatedKFold(n_splits=5, n_repeats=30), "
                "RepeatedStratifiedKFold(n_splits=5, n_repeats=30) for "
                "classifiers, or ShuffleSplit(n_splits=150, test_size=0.2)."
            )
        if test != "auto":
            raise ValueError(
                "test must be 'auto' when estimators are given: the splitter "
                "decides which test is valid. Use test= only together with "
                "diff_AB."
            )
        diff, which, default_rho = _build_diff_AB(
            estimator_1,
            estimator_2,
            X,
            y,
            cv=cv,
            scoring=scoring,
            stratify=stratify,
            random_state=random_state,
        )
        if fall_back_rho is None:
            fall_back_rho = default_rho

    run = sha_test if which == "sha" else sharp_test
    z, p = run(diff, fall_back_rho=fall_back_rho, mode=mode)
    return SharpTestResult(
        statistic=float(z),
        pvalue=float(p),
        test=which,
        mode=mode,
        fall_back_rho=None if fall_back_rho is None else float(fall_back_rho),
        diff_AB=diff,
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


def _warn_if_splitter_seeded(splitter) -> None:
    """Warn when a repeated / Monte-Carlo splitter carries a ``random_state``
    that :func:`sharp_cross_val_test` will not use."""
    if getattr(splitter, "random_state", None) is not None:
        warnings.warn(
            f"The random_state of the {type(splitter).__name__} passed as cv is "
            "ignored: only its n_splits, n_repeats, test_size and train_size "
            "are read, and every split is drawn from the random_state given to "
            "sharp_cross_val_test. Pass random_state= to sharp_cross_val_test "
            "instead.",
            UserWarning,
            stacklevel=5,
        )


def _resolve_scheme(cv, y, is_clf: bool, random_state=None) -> _Scheme:
    splitter = check_cv(cv, y, classifier=is_clf)

    if isinstance(splitter, _GROUP_SPLITTERS):
        raise ValueError(
            f"{type(splitter).__name__} is not supported: sharp_cross_val_test "
            "has no groups argument, and the half-split would place samples of "
            "one group in both halves. Use KFold, StratifiedKFold, RepeatedKFold, "
            "RepeatedStratifiedKFold, ShuffleSplit or StratifiedShuffleSplit."
        )

    # RepeatedKFold / RepeatedStratifiedKFold and user subclasses of
    # sklearn's _RepeatedSplits: one fresh shuffled K-fold per repetition.
    if all(hasattr(splitter, a) for a in ("n_repeats", "cv", "cvargs")):
        _warn_if_splitter_seeded(splitter)
        base, kwargs = splitter.cv, dict(splitter.cvargs)
        n_repeats = int(splitter.n_repeats)

        def make_inner(rng):
            return base(shuffle=True, random_state=rng.randint(_MAX_SEED), **kwargs)

        return _Scheme("repeated_kfold", n_repeats, make_inner)

    # ShuffleSplit / StratifiedShuffleSplit: one train/test split per half
    # per repetition, with the user's test_size / train_size.
    if isinstance(splitter, (ShuffleSplit, StratifiedShuffleSplit)):
        _warn_if_splitter_seeded(splitter)
        cls, n_repeats = type(splitter), int(splitter.n_splits)
        test_size, train_size = splitter.test_size, splitter.train_size

        def make_inner(rng):
            return cls(
                n_splits=1,
                test_size=test_size,
                train_size=train_size,
                random_state=rng.randint(_MAX_SEED),
            )

        return _Scheme("monte_carlo", n_repeats, make_inner)

    # Anything else would be run once inside each half of a single
    # half-split, which only the SHA test can analyse.
    _require_sha(
        f"cv={type(splitter).__name__} runs a single cross-validation inside "
        "one pair of halves, with no repetition. "
    )

    # Unreachable while SHA is switched off. A shuffling splitter without
    # its own seed defeats the caller's random_state.
    if (
        random_state is not None
        and getattr(splitter, "shuffle", False)
        and getattr(splitter, "random_state", None) is None
    ):
        warnings.warn(
            f"cv={type(splitter).__name__} has shuffle=True but no random_state, "
            "so its folds differ on every call and the result is not "
            "reproducible even though random_state was given to "
            "sharp_cross_val_test. Set random_state on the splitter as well.",
            UserWarning,
            stacklevel=4,
        )
    return _Scheme("kfold", 1, lambda rng: splitter)


def _n_rows(a) -> int:
    return a.shape[0] if hasattr(a, "shape") else len(a)


def _take(a, idx):
    """Rows ``idx`` of ``a``, keeping pandas and sparse containers as they are."""
    if hasattr(a, "iloc"):
        return a.iloc[idx]
    return a[idx]


def _build_diff_AB(
    estimator_1,
    estimator_2,
    X,
    y,
    *,
    cv,
    scoring=None,
    stratify: bool | None = None,
    random_state=None,
):
    """Run the split-half procedure.

    Returns ``(diff_AB, test, default_fall_back_rho)`` where ``test`` is
    ``'sharp'`` or ``'sha'``.
    """
    if _n_rows(X) != _n_rows(y):
        raise ValueError(
            "X and y must have the same number of rows; got "
            f"{_n_rows(X)} and {_n_rows(y)}"
        )
    # Lists become arrays; pandas objects and sparse matrices are kept.
    X, y = indexable(X, y)

    is_clf = stratify if isinstance(stratify, bool) else is_classifier(estimator_1)
    rng = check_random_state(random_state)
    scheme = _resolve_scheme(cv, y, is_clf, random_state=random_state)
    if scheme.kind != "kfold" and scheme.n_repeats < 2:
        raise ValueError(
            "SHARP needs at least 2 repetitions; got "
            f"{scheme.n_repeats} (n_repeats / n_splits of the splitter)."
        )
    scorer = check_scoring(estimator_1, scoring=scoring)

    repetitions = []  # one [diffs_half_A, diffs_half_B] per repetition
    test_fractions = []
    for _ in range(scheme.n_repeats):
        X_A, X_B, y_A, y_B = train_test_split(
            X,
            y,
            test_size=0.5,
            stratify=y if is_clf else None,
            random_state=rng.randint(_MAX_SEED),
        )
        inner = scheme.make_inner(rng)
        halves = []
        for X_h, y_h in ((X_A, y_A), (X_B, y_B)):
            diffs = []
            for train_idx, test_idx in inner.split(X_h, y_h):
                X_tr, y_tr = _take(X_h, train_idx), _take(y_h, train_idx)
                X_te, y_te = _take(X_h, test_idx), _take(y_h, test_idx)
                a = clone(estimator_1).fit(X_tr, y_tr)
                b = clone(estimator_2).fit(X_tr, y_tr)
                diffs.append(scorer(a, X_te, y_te) - scorer(b, X_te, y_te))
                test_fractions.append(len(test_idx) / _n_rows(y_h))
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
        # Single repetition: keep the per-fold rows. Unreachable while SHA
        # is switched off; _resolve_scheme raises first.
        diff = np.column_stack(repetitions[0])
        return diff, "sha", 1.0 / (2 * n_inner)

    # Repeated schemes: one row per repetition, folds averaged within a half.
    diff = np.array(
        [[np.mean(half_A), np.mean(half_B)] for half_A, half_B in repetitions]
    )
    if scheme.kind == "monte_carlo":
        return diff, "sharp", float(np.mean(test_fractions)) / 2.0
    return diff, "sharp", 1.0 / (2 * n_inner)
