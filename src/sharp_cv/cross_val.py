"""sklearn-style entry point that runs the split-half procedure end to end.

:func:`sharp_cross_val_test` takes two sklearn estimators and ``(X, y)``,
runs the split-half procedure and hands the resulting paired differences
to :func:`~sharp_cv.sharp_test`. To test paired differences that you
computed yourself, call :func:`~sharp_cv.sharp_test` directly.

One *repetition* of the procedure is:

1. divide the data at random into two disjoint halves A and B (stratified
   by class for classifiers);
2. inside each half, run the inner cross-validation, fitting both
   estimators on the same training folds and scoring them on the same
   test folds;
3. reduce each half to one mean score per estimator and one mean
   difference, the mean over ``score_1 - score_2``.

``cv`` is required, and sets both the inner cross-validation and the
number of repetitions:

* ``RepeatedKFold(n_splits=K, n_repeats=J)`` or
  ``RepeatedStratifiedKFold(...)``: J repetitions, K-fold inside each half
  with the folds averaged. ``diff_AB`` is ``[J, 2]``. Around ``J = 30`` is
  a reasonable starting point.
* ``ShuffleSplit(n_splits=J, test_size=t)`` or
  ``StratifiedShuffleSplit(...)``: J repetitions, one train/test split
  inside each half. ``diff_AB`` is ``[J, 2]``. ``n_splits`` counts
  repetitions here, not inner folds; around ``J = 150`` is a reasonable
  starting point.
* ``KFold(K)``, ``StratifiedKFold(K)`` or an int ``K``: a single
  repetition, which only the SHA test can analyse. Rejected while SHA is
  switched off; see :mod:`sharp_cv._engine`.

For the repeated and Monte-Carlo splitters only ``n_splits``,
``n_repeats``, ``test_size``, ``train_size`` and ``random_state`` are
read; their ``split`` method is never called. Every split is drawn from
one random stream. That stream is seeded by the ``random_state`` given to
:func:`sharp_cross_val_test` or, when that is ``None``, by the splitter's
own ``random_state``, so seeding either one gives a reproducible result.
A plain ``KFold`` / ``StratifiedKFold`` is used as given. Group-aware
splitters such as ``GroupKFold`` are rejected: there is no ``groups``
argument, and the half-split would place samples of one group in both
halves.

Repetitions are independent of each other and run in parallel with
``n_jobs``, as in :func:`sklearn.model_selection.cross_val_score`. The
seeds of all repetitions are drawn before any of them runs, so the splits,
the fits and their order do not depend on ``n_jobs``. Only the last bits
can move, because joblib pins BLAS to one thread inside its workers.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, NamedTuple

import numpy as np
from joblib import Parallel, delayed
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

from sharp_cv._engine import _SHA_AVAILABLE, VALID_MODES, sha_test, sharp_test

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
        statistic: z statistic. Positive when ``estimator_1`` scored higher
            on average, negative when ``estimator_2`` did.
        pvalue: two-sided p-value.
        mean_diff: mean of ``diff_AB``, the estimated performance
            difference (``estimator_1`` minus ``estimator_2``) in the units
            of the score.
        test: the test that produced the result, always ``'sharp'`` while
            SHA is switched off.
        mode: estimator mode that was used.
        fall_back_rho: fallback correlation that was in effect. Read by
            ``'mm'`` and ``'mmc'`` only; the other modes ignore it.
        diff_AB: ``[J, 2]`` paired differences that were tested, one row
            per repetition, column 0 from half A and column 1 from half B.
        score_1_AB: ``[J, 2]`` mean score of ``estimator_1`` in half A and
            in half B of each repetition.
        score_2_AB: the same for ``estimator_2``. ``diff_AB`` equals
            ``score_1_AB - score_2_AB`` up to floating-point rounding.
    """

    statistic: float
    pvalue: float
    mean_diff: float
    test: str
    mode: str
    fall_back_rho: float
    diff_AB: np.ndarray
    score_1_AB: np.ndarray
    score_2_AB: np.ndarray

    def __repr__(self) -> str:
        # Arrays are summarised by shape; their contents are one attribute
        # access away and would otherwise fill a notebook cell.
        parts = []
        for name, value in zip(self._fields, self):
            if isinstance(value, np.ndarray):
                shown = f"<array of shape {value.shape}>"
            else:
                shown = repr(value)
            parts.append(f"{name}={shown}")
        return f"{type(self).__name__}({', '.join(parts)})"


def sharp_cross_val_test(
    estimator_1,
    estimator_2,
    X,
    y,
    *,
    cv,
    scoring=None,
    stratify: bool | None = None,
    mode: str = "st",
    fall_back_rho: float | None = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    random_state=None,
) -> SharpTestResult:
    """Compare two estimators with the SHARP split-half test.

    Args:
        estimator_1, estimator_2: sklearn-compatible estimators. Both are
            scored with the same scorer, so with ``scoring=None`` their
            ``score`` methods must measure the same thing; a classifier
            and a regressor are rejected in that case. The tested
            difference is ``estimator_1`` minus ``estimator_2``. A
            ``GridSearchCV``, or a pipeline containing one, is refit inside
            every training fold, which gives nested cross-validation.
        X, y: feature matrix and target. NumPy arrays, pandas objects and
            SciPy sparse matrices are accepted; rows are selected with
            ``.iloc`` for pandas, so column names reach the estimators.
        cv: sklearn splitter selecting the inner cross-validation and the
            number of split-half repetitions; see the module docstring.
            Required, because the repetition count is a choice: around 30
            for ``RepeatedKFold`` / ``RepeatedStratifiedKFold``, around 150
            for ``ShuffleSplit`` / ``StratifiedShuffleSplit``. Single-run
            splitters (an int, ``KFold``, ``StratifiedKFold``) have no
            repetition and are rejected, as are group-aware splitters.
        scoring: sklearn scoring string or callable. ``None`` uses the
            estimators' ``score`` method.
        stratify: ``None`` stratifies the half-split when ``estimator_1``
            is a classifier; ``True`` / ``False`` force the choice.
        mode: estimator mode, one of ``'mm'``, ``'mmc'``, ``'ml'``,
            ``'rml'``, ``'lrt'``, ``'st'`` (default, recommended). See
            :mod:`sharp_cv._engine`.
        fall_back_rho: correlation used by ``'mm'`` / ``'mmc'`` when the
            estimated variance falls below its minimum; ignored by every
            other mode, including the default. The default is
            ``1 / (2 * K)`` for K-fold inner schemes and ``test_size / 2``
            for Monte-Carlo schemes, where ``test_size`` is the fraction
            actually used inside each half. Halving is what expresses these
            as a fraction of the full dataset, since an inner test set is
            drawn from a half.
        n_jobs: number of repetitions to run in parallel, passed to
            :class:`joblib.Parallel` as in ``cross_val_score``. ``None``
            means one, ``-1`` all processors. The result does not depend on
            it.
        verbose: verbosity of the progress output, passed to
            :class:`joblib.Parallel`. ``0`` is silent.
        random_state: seed of the single random stream that draws the
            half-splits and, for repeated and Monte-Carlo schemes, the
            inner splits. When ``None``, the ``random_state`` of a
            ``RepeatedKFold``, ``RepeatedStratifiedKFold``, ``ShuffleSplit``
            or ``StratifiedShuffleSplit`` passed as ``cv`` is used instead,
            so seeding the splitter alone is enough. When both are given,
            this one wins and a warning is raised. A plain ``KFold`` /
            ``StratifiedKFold`` object is used as given and keeps its own
            ``shuffle`` / ``random_state``.

    Returns:
        :class:`SharpTestResult`.
    """
    if mode not in VALID_MODES:
        hint = " mode='all' was removed; call once per mode." if mode == "all" else ""
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}.{hint}")

    proc = _build_diff_AB(
        estimator_1,
        estimator_2,
        X,
        y,
        cv=cv,
        scoring=scoring,
        stratify=stratify,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
    )
    if fall_back_rho is None:
        fall_back_rho = proc.fall_back_rho

    run = sha_test if proc.test == "sha" else sharp_test
    z, p = run(proc.diff_AB, fall_back_rho=fall_back_rho, mode=mode)
    return SharpTestResult(
        statistic=float(z),
        pvalue=float(p),
        mean_diff=float(np.mean(proc.diff_AB)),
        test=proc.test,
        mode=mode,
        fall_back_rho=float(fall_back_rho),
        diff_AB=proc.diff_AB,
        score_1_AB=proc.score_1_AB,
        score_2_AB=proc.score_2_AB,
    )


# ---------------------------------------------------------------------------
# Procedure
# ---------------------------------------------------------------------------


class _Scheme(NamedTuple):
    """How ``cv`` maps onto the split-half procedure.

    ``kind`` is ``'repeated_kfold'``, ``'monte_carlo'`` or ``'kfold'``.
    ``n_repeats`` is the number of independent half-splits.
    ``make_inner(rng)`` returns the splitter run inside each half of one
    repetition. ``random_state`` seeds the stream that ``rng`` is drawn
    from; see :func:`_effective_random_state`.
    """

    kind: str
    n_repeats: int
    make_inner: Callable[[np.random.RandomState], Any]
    random_state: Any


class _Procedure(NamedTuple):
    """Output of :func:`_build_diff_AB`.

    ``test`` is ``'sharp'`` or ``'sha'``; ``fall_back_rho`` is the default
    for that scheme. The three arrays have the same shape.
    """

    diff_AB: np.ndarray
    score_1_AB: np.ndarray
    score_2_AB: np.ndarray
    test: str
    fall_back_rho: float


def _effective_random_state(splitter, random_state):
    """Seed of the single random stream that drives every split.

    ``random_state`` given to :func:`sharp_cross_val_test` wins. When it is
    ``None`` the splitter's own ``random_state`` is used instead, so the
    usual sklearn habit of seeding the splitter gives a reproducible
    result. When both are given the splitter's is ignored, with a warning.
    """
    own = getattr(splitter, "random_state", None)
    if own is None:
        return random_state
    if random_state is None:
        return own
    warnings.warn(
        f"Both the {type(splitter).__name__} passed as cv and "
        "sharp_cross_val_test were given a random_state. The splitter's is "
        "ignored: every split is drawn from the random_state given to "
        "sharp_cross_val_test. Pass only one of the two.",
        UserWarning,
        stacklevel=5,
    )
    return random_state


def _resolve_scheme(cv, y, is_clf: bool, random_state=None) -> _Scheme:
    splitter = check_cv(cv, y, classifier=is_clf)

    if isinstance(splitter, _GROUP_SPLITTERS):
        raise ValueError(
            f"{type(splitter).__name__} is not supported: sharp_cross_val_test "
            "has no groups argument, and the half-split would place samples of "
            "one group in both halves. Use RepeatedKFold, "
            "RepeatedStratifiedKFold, ShuffleSplit or StratifiedShuffleSplit."
        )

    # RepeatedKFold / RepeatedStratifiedKFold and user subclasses of
    # sklearn's _RepeatedSplits: one fresh shuffled K-fold per repetition.
    if all(hasattr(splitter, a) for a in ("n_repeats", "cv", "cvargs")):
        seed = _effective_random_state(splitter, random_state)
        base, kwargs = splitter.cv, dict(splitter.cvargs)
        n_repeats = int(splitter.n_repeats)

        def make_inner(rng):
            return base(shuffle=True, random_state=rng.randint(_MAX_SEED), **kwargs)

        return _Scheme("repeated_kfold", n_repeats, make_inner, seed)

    # ShuffleSplit / StratifiedShuffleSplit: one train/test split per half
    # per repetition, with the user's test_size / train_size.
    if isinstance(splitter, (ShuffleSplit, StratifiedShuffleSplit)):
        seed = _effective_random_state(splitter, random_state)
        cls, n_repeats = type(splitter), int(splitter.n_splits)
        test_size, train_size = splitter.test_size, splitter.train_size

        def make_inner(rng):
            return cls(
                n_splits=1,
                test_size=test_size,
                train_size=train_size,
                random_state=rng.randint(_MAX_SEED),
            )

        return _Scheme("monte_carlo", n_repeats, make_inner, seed)

    # Anything else would be run once inside each half of a single
    # half-split, which only the SHA test can analyse.
    if not _SHA_AVAILABLE:
        raise NotImplementedError(
            f"cv={type(splitter).__name__} runs a single cross-validation "
            "inside one pair of halves, with no repetition, which this release "
            "cannot test. Pass a repeated splitter such as "
            "RepeatedKFold(n_splits=5, n_repeats=30), "
            "RepeatedStratifiedKFold(n_splits=5, n_repeats=30) for classifiers, "
            "or ShuffleSplit(n_splits=150, test_size=0.2)."
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
    return _Scheme("kfold", 1, lambda rng: splitter, random_state)


def _n_rows(a) -> int:
    return a.shape[0] if hasattr(a, "shape") else len(a)


def _take(a, idx):
    """Rows ``idx`` of ``a``, keeping pandas and sparse containers as they are."""
    if hasattr(a, "iloc"):
        return a.iloc[idx]
    return a[idx]


def _one_repetition(estimator_1, estimator_2, X, y, scorer, half_seed, inner, stratify):
    """One repetition: split into halves, cross-validate inside each.

    Returns ``(score_1, score_2, diff, test_fractions)``. The first three
    are ``[values_half_A, values_half_B]`` with one value per inner split;
    ``test_fractions`` is the fraction of a half held out by each inner
    test set, in the order the splits were run.
    """
    X_A, X_B, y_A, y_B = train_test_split(
        X,
        y,
        test_size=0.5,
        stratify=y if stratify else None,
        random_state=half_seed,
    )
    score_1, score_2, diff, test_fractions = [], [], [], []
    for X_h, y_h in ((X_A, y_A), (X_B, y_B)):
        s1, s2, d = [], [], []
        for train_idx, test_idx in inner.split(X_h, y_h):
            X_tr, y_tr = _take(X_h, train_idx), _take(y_h, train_idx)
            X_te, y_te = _take(X_h, test_idx), _take(y_h, test_idx)
            a = clone(estimator_1).fit(X_tr, y_tr)
            b = clone(estimator_2).fit(X_tr, y_tr)
            score_a, score_b = scorer(a, X_te, y_te), scorer(b, X_te, y_te)
            s1.append(score_a)
            s2.append(score_b)
            d.append(score_a - score_b)
            test_fractions.append(len(test_idx) / _n_rows(y_h))
        score_1.append(s1)
        score_2.append(s2)
        diff.append(d)
    if len(diff[0]) != len(diff[1]):
        raise RuntimeError(
            "The two halves produced different numbers of inner splits "
            f"({len(diff[0])} vs {len(diff[1])}); use a splitter with "
            "a fixed n_splits."
        )
    return score_1, score_2, diff, test_fractions


def _build_diff_AB(
    estimator_1,
    estimator_2,
    X,
    y,
    *,
    cv,
    scoring=None,
    stratify: bool | None = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    random_state=None,
) -> _Procedure:
    """Run the split-half procedure; see :class:`_Procedure`."""
    if _n_rows(X) != _n_rows(y):
        raise ValueError(
            "X and y must have the same number of rows; got "
            f"{_n_rows(X)} and {_n_rows(y)}"
        )
    if scoring is None and is_classifier(estimator_1) != is_classifier(estimator_2):
        kind_1 = "classifier" if is_classifier(estimator_1) else "regressor"
        kind_2 = "classifier" if is_classifier(estimator_2) else "regressor"
        raise ValueError(
            f"estimator_1 is a {kind_1} and estimator_2 is a {kind_2}. With "
            "scoring=None each is scored by its own score method, which would "
            "compare two different metrics. Pass scoring= explicitly."
        )
    # Lists become arrays; pandas objects and sparse matrices are kept.
    X, y = indexable(X, y)

    is_clf = stratify if isinstance(stratify, bool) else is_classifier(estimator_1)
    scheme = _resolve_scheme(cv, y, is_clf, random_state=random_state)
    if scheme.kind != "kfold" and scheme.n_repeats < 2:
        raise ValueError(
            "SHARP needs at least 2 repetitions; got "
            f"{scheme.n_repeats} (n_repeats / n_splits of the splitter)."
        )
    scorer = check_scoring(estimator_1, scoring=scoring)

    # Every seed is drawn here, in repetition order, before any fitting, so
    # the result is the same whatever n_jobs is.
    rng = check_random_state(scheme.random_state)
    tasks = []
    for _ in range(scheme.n_repeats):
        half_seed = rng.randint(_MAX_SEED)
        tasks.append((half_seed, scheme.make_inner(rng)))

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_one_repetition)(
            estimator_1, estimator_2, X, y, scorer, half_seed, inner, is_clf
        )
        for half_seed, inner in tasks
    )

    n_inner = len(results[0][2][0])

    if scheme.kind == "kfold":
        # Single repetition: keep the per-fold rows. Unreachable while SHA
        # is switched off; _resolve_scheme raises first.
        score_1, score_2, diff, _ = results[0]
        return _Procedure(
            np.column_stack(diff),
            np.column_stack(score_1),
            np.column_stack(score_2),
            "sha",
            1.0 / (2 * n_inner),
        )

    # Repeated schemes: one row per repetition, folds averaged within a half.
    def half_means(values):
        return np.array(
            [[np.mean(half_A), np.mean(half_B)] for half_A, half_B in values]
        )

    diff = half_means(r[2] for r in results)
    score_1 = half_means(r[0] for r in results)
    score_2 = half_means(r[1] for r in results)
    if scheme.kind == "monte_carlo":
        test_fractions = [f for r in results for f in r[3]]
        rho = float(np.mean(test_fractions)) / 2.0
        return _Procedure(diff, score_1, score_2, "sharp", rho)
    return _Procedure(diff, score_1, score_2, "sharp", 1.0 / (2 * n_inner))
