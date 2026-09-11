"""End-to-end tests for sharp_cv.sharp_cross_val_test."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import check_scoring
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)

from sharp_cv import SharpTestResult, sharp_cross_val_test, sharp_test
from sharp_cv.cross_val import _build_diff_AB, _resolve_scheme

MAX_SEED = 2**31 - 1


def _check_result(r: SharpTestResult, test: str, shape) -> None:
    assert isinstance(r, SharpTestResult)
    assert r.test == test
    assert r.diff_AB.shape == shape
    assert np.isfinite(r.statistic)
    assert 0.0 <= r.pvalue <= 1.0


def _regression():
    return make_regression(n_samples=200, noise=1.0, random_state=0)


def _classification(n_classes=2):
    return make_classification(
        n_samples=400,
        n_features=10,
        n_informative=5,
        n_classes=n_classes,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Dispatch and shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cv", [5, KFold(4), KFold(3, shuffle=True), StratifiedKFold(5)])
def test_single_run_splitters_are_rejected_while_sha_is_off(cv):
    X, y = _regression()
    with pytest.raises(NotImplementedError, match="SHA test is not available"):
        sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, cv=cv, random_state=0)


def test_single_run_rejection_names_the_splitter():
    _, y = _classification(n_classes=3)
    with pytest.raises(NotImplementedError, match="cv=StratifiedKFold runs a single"):
        _resolve_scheme(5, y, is_clf=True)


def test_single_run_splitter_rejected_before_any_fitting():
    X, y = _regression()
    _Recorder.fits = []
    with pytest.raises(NotImplementedError):
        sharp_cross_val_test(_Recorder(), _Recorder(), X, y, cv=5, random_state=0)
    assert _Recorder.fits == []


def test_repeated_kfold_gives_one_row_per_repetition():
    X, y = _regression()
    r = sharp_cross_val_test(
        Ridge(0.1),
        Ridge(10.0),
        X,
        y,
        cv=RepeatedKFold(n_splits=4, n_repeats=6),
        random_state=0,
    )
    _check_result(r, "sharp", (6, 2))
    assert r.fall_back_rho == pytest.approx(1 / 8)  # 1 / (2K), K = 4


def test_repeated_stratified_kfold():
    X, y = _classification()
    r = sharp_cross_val_test(
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=10, random_state=0),
        X,
        y,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=4),
        random_state=0,
    )
    _check_result(r, "sharp", (4, 2))
    assert r.fall_back_rho == pytest.approx(1 / 10)


@pytest.mark.parametrize(
    "test_size, expected_rho",
    [
        (0.2, 0.1),  # float fraction of the 100-sample half
        (None, 0.05),  # sklearn default 0.1
        (20, 0.1),  # absolute count: 20 of 100
    ],
)
def test_shuffle_split_fall_back_rho_from_actual_test_fraction(test_size, expected_rho):
    X, y = _regression()  # 200 samples -> halves of 100
    diff, test, rho = _build_diff_AB(
        Ridge(0.1),
        Ridge(10.0),
        X,
        y,
        cv=ShuffleSplit(n_splits=6, test_size=test_size),
        random_state=0,
    )
    assert test == "sharp"
    assert diff.shape == (6, 2)
    assert rho == pytest.approx(expected_rho)


def test_stratified_shuffle_split():
    X, y = _classification()
    r = sharp_cross_val_test(
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=10, random_state=0),
        X,
        y,
        cv=StratifiedShuffleSplit(n_splits=8, test_size=0.25),
        random_state=0,
    )
    _check_result(r, "sharp", (8, 2))
    assert r.fall_back_rho == pytest.approx(0.125)


def test_scoring_string():
    X, y = _classification()
    r = sharp_cross_val_test(
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=10, random_state=0),
        X,
        y,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
        scoring="accuracy",
        random_state=0,
    )
    _check_result(r, "sharp", (3, 2))


def test_repeated_kfold_equals_sharp_test_on_returned_diff():
    X, y = _regression()
    r = sharp_cross_val_test(
        Ridge(0.1),
        Ridge(10.0),
        X,
        y,
        cv=RepeatedKFold(n_splits=5, n_repeats=6),
        random_state=0,
    )
    z, p = sharp_test(r.diff_AB, fall_back_rho=1 / 10, mode="st")
    assert r.statistic == pytest.approx(z)
    assert r.pvalue == pytest.approx(p)


# ---------------------------------------------------------------------------
# The procedure itself
# ---------------------------------------------------------------------------


class _Recorder(BaseEstimator, RegressorMixin):
    """Predicts the training mean and records the row ids it was fit on.
    Row ids are read from column 0 of X."""

    fits: list = []

    def fit(self, X, y):
        type(self).fits.append(frozenset(int(v) for v in X[:, 0]))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_)


def test_repeated_kfold_redraws_halves_every_repetition():
    n, K, J = 60, 3, 4
    rng = np.random.default_rng(0)
    X = np.column_stack([np.arange(n), rng.normal(size=n)])
    y = 2.0 * X[:, 1] + rng.normal(size=n)

    _Recorder.fits = []
    _build_diff_AB(
        _Recorder(),
        _Recorder(),
        X,
        y,
        cv=RepeatedKFold(n_splits=K, n_repeats=J),
        random_state=0,
    )
    fits = _Recorder.fits
    assert len(fits) == J * 2 * K * 2  # repetitions x halves x folds x models

    first_halves = []
    for j in range(J):
        halves = []
        for h in range(2):
            start = ((j * 2 + h) * K) * 2
            block = fits[start:start + 2 * K]
            # Both models are fit on identical training rows, fold by fold.
            assert block[0::2] == block[1::2]
            # Union of the K training sets is the whole half.
            halves.append(frozenset().union(*block[0::2]))
        assert halves[0].isdisjoint(halves[1])
        assert len(halves[0] | halves[1]) == n
        assert len(halves[0]) == n // 2
        first_halves.append(halves[0])
    # A fresh half-split on every repetition.
    assert len(set(first_halves)) == J


def test_monte_carlo_uses_one_split_per_half_per_repetition():
    n, J = 60, 5
    rng = np.random.default_rng(1)
    X = np.column_stack([np.arange(n), rng.normal(size=n)])
    y = 2.0 * X[:, 1] + rng.normal(size=n)

    _Recorder.fits = []
    diff, test, rho = _build_diff_AB(
        _Recorder(),
        _Recorder(),
        X,
        y,
        cv=ShuffleSplit(n_splits=J, test_size=0.2),
        random_state=0,
    )
    assert test == "sharp"
    assert diff.shape == (J, 2)
    assert rho == pytest.approx(0.1)
    fits = _Recorder.fits
    assert len(fits) == J * 2 * 2  # repetitions x halves x models
    # Training set is 80% of a 30-sample half.
    assert all(len(s) == 24 for s in fits)
    # Halves of one repetition are disjoint; halves change across repetitions.
    half_A_train = [fits[4 * j] for j in range(J)]
    half_B_train = [fits[4 * j + 2] for j in range(J)]
    assert all(a.isdisjoint(b) for a, b in zip(half_A_train, half_B_train))
    assert len(set(half_A_train)) == J


def test_repeated_kfold_rows_are_fold_means_of_fresh_halves():
    """Reproduce the procedure by hand, including the random stream."""
    X, y = _regression()
    K, J = 5, 3
    # The splitter's own seed is ignored (and warned about); only the
    # random_state passed to the procedure drives the streams below.
    with pytest.warns(UserWarning, match="random_state of the RepeatedKFold"):
        diff, _, _ = _build_diff_AB(
            Ridge(0.1),
            Ridge(10.0),
            X,
            y,
            cv=RepeatedKFold(n_splits=K, n_repeats=J, random_state=123),
            random_state=0,
        )

    rng = np.random.RandomState(0)
    scorer = check_scoring(Ridge(0.1), scoring=None)
    expected = []
    for _ in range(J):
        X_A, X_B, y_A, y_B = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=rng.randint(MAX_SEED),
        )
        inner = KFold(n_splits=K, shuffle=True, random_state=rng.randint(MAX_SEED))
        row = []
        for X_h, y_h in ((X_A, y_A), (X_B, y_B)):
            d = []
            for tr, te in inner.split(X_h, y_h):
                a = clone(Ridge(0.1)).fit(X_h[tr], y_h[tr])
                b = clone(Ridge(10.0)).fit(X_h[tr], y_h[tr])
                d.append(scorer(a, X_h[te], y_h[te]) - scorer(b, X_h[te], y_h[te]))
            row.append(np.mean(d))
        expected.append(row)
    np.testing.assert_allclose(diff, np.asarray(expected), rtol=1e-12)


def test_reproducible_with_same_random_state():
    X, y = _regression()
    kw = dict(cv=RepeatedKFold(n_splits=5, n_repeats=3), random_state=7)
    r1 = sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, **kw)
    r2 = sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, **kw)
    assert r1.statistic == r2.statistic
    assert r1.pvalue == r2.pvalue
    np.testing.assert_array_equal(r1.diff_AB, r2.diff_AB)


def test_stratify_override():
    X, y = _classification()
    r = sharp_cross_val_test(
        Ridge(0.1),
        Ridge(10.0),
        X,
        y,
        cv=RepeatedKFold(n_splits=4, n_repeats=3),
        stratify=True,
        random_state=0,
    )
    _check_result(r, "sharp", (3, 2))


# ---------------------------------------------------------------------------
# Seeds carried by the cv splitter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_cv",
    [
        lambda rs: RepeatedKFold(n_splits=3, n_repeats=3, random_state=rs),
        lambda rs: ShuffleSplit(n_splits=3, test_size=0.2, random_state=rs),
    ],
)
def test_repeated_and_monte_carlo_splitter_seed_is_ignored_with_warning(make_cv):
    X, y = _regression()
    results = []
    for rs in (0, 999):
        with pytest.warns(UserWarning, match="random_state of the .* is ignored"):
            results.append(sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, cv=make_cv(rs), random_state=0))
    np.testing.assert_array_equal(results[0].diff_AB, results[1].diff_AB)


def test_unseeded_or_self_seeded_splitters_do_not_warn():
    X, y = _regression()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for cv in (
                RepeatedKFold(n_splits=3, n_repeats=2),
                ShuffleSplit(n_splits=2, test_size=0.2),
        ):
            sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, cv=cv, random_state=0)


def test_group_splitter_rejected():
    X, y = _regression()
    with pytest.raises(ValueError, match="GroupKFold is not supported"):
        sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, cv=GroupKFold(3), random_state=0)


# ---------------------------------------------------------------------------
# Input containers
# ---------------------------------------------------------------------------


def test_dataframe_input_matches_ndarray_and_keeps_column_names():
    pd = pytest.importorskip("pandas")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _regression()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    kw = dict(cv=RepeatedKFold(n_splits=3, n_repeats=3), random_state=0)
    r_np = sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, **kw)
    r_df = sharp_cross_val_test(Ridge(0.1), Ridge(10.0), df, pd.Series(y), **kw)
    np.testing.assert_array_equal(r_np.diff_AB, r_df.diff_AB)

    # Selecting columns by name only works if the DataFrame reaches the
    # pipeline intact.
    ct = ColumnTransformer([("sc", StandardScaler(), ["f0", "f1", "f2"])])
    r = sharp_cross_val_test(
        make_pipeline(ct, Ridge(0.1)),
        make_pipeline(ct, Ridge(10.0)),
        df,
        y,
        cv=RepeatedKFold(n_splits=3, n_repeats=2),
        random_state=0,
    )
    _check_result(r, "sharp", (2, 2))


def test_sparse_input():
    sp = pytest.importorskip("scipy.sparse")
    X, y = _regression()
    r = sharp_cross_val_test(
        Ridge(0.1),
        Ridge(10.0),
        sp.csr_matrix(X),
        y,
        cv=RepeatedKFold(n_splits=3, n_repeats=2),
        random_state=0,
    )
    _check_result(r, "sharp", (2, 2))


# ---------------------------------------------------------------------------
# diff_AB path
# ---------------------------------------------------------------------------


def _diff(seed=0, n=8):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal([0.05, 0.05], [[0.01, 0.0], [0.0, 0.01]], size=n)


def test_diff_AB_sharp_matches_sharp_test():
    diff = _diff()
    z, p = sharp_test(diff, fall_back_rho=1 / 10, mode="st")
    r = sharp_cross_val_test(diff_AB=diff, test="sharp", fall_back_rho=1 / 10, mode="st")
    assert r.statistic == pytest.approx(z)
    assert r.pvalue == pytest.approx(p)
    assert r.test == "sharp"
    assert r.fall_back_rho == pytest.approx(1 / 10)


def test_diff_AB_sha_is_switched_off():
    with pytest.raises(NotImplementedError, match="SHA test is not available"):
        sharp_cross_val_test(diff_AB=_diff(), test="sha", fall_back_rho=1 / 16, mode="mm")


def test_diff_AB_requires_test():
    with pytest.raises(ValueError, match="test cannot be inferred"):
        sharp_cross_val_test(diff_AB=_diff(), fall_back_rho=0.1)


def test_diff_AB_fall_back_rho_required_only_for_mm_and_mmc():
    for mode in ("mm", "mmc"):
        with pytest.raises(ValueError, match="fall_back_rho is required"):
            sharp_cross_val_test(diff_AB=_diff(), test="sharp", mode=mode)
    r = sharp_cross_val_test(diff_AB=_diff(), test="sharp")  # default 'st'
    z, p = sharp_test(_diff())
    assert r.fall_back_rho is None
    assert r.statistic == pytest.approx(z)
    assert r.pvalue == pytest.approx(p)


def test_diff_AB_shape_validation():
    with pytest.raises(ValueError, match=r"shape \[n, 2\]"):
        sharp_cross_val_test(diff_AB=np.zeros((5, 3)), test="sharp", fall_back_rho=0.1)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_missing_inputs_raises():
    with pytest.raises(ValueError, match="Provide either"):
        sharp_cross_val_test()


def test_cv_is_required_with_estimators():
    X, y = _regression()
    with pytest.raises(ValueError, match="cv is required"):
        sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, random_state=0)


def test_cv_is_not_needed_with_diff_AB():
    r = sharp_cross_val_test(diff_AB=_diff(), test="sharp")
    assert r.test == "sharp"


def test_test_override_rejected_with_estimators():
    X, y = _regression()
    with pytest.raises(ValueError, match="test must be 'auto'"):
        sharp_cross_val_test(
            Ridge(0.1),
            Ridge(10.0),
            X,
            y,
            cv=RepeatedKFold(n_splits=5, n_repeats=3),
            test="sharp",
        )


def test_invalid_test_kwarg_raises():
    with pytest.raises(ValueError, match="test must be one of"):
        sharp_cross_val_test(diff_AB=_diff(), test="bogus", fall_back_rho=0.1)


def test_mode_all_rejected_before_any_fitting():
    X, y = _regression()
    with pytest.raises(ValueError, match="mode='all' was removed"):
        sharp_cross_val_test(
            Ridge(0.1),
            Ridge(10.0),
            X,
            y,
            cv=RepeatedKFold(n_splits=5, n_repeats=3),
            mode="all",
        )


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        sharp_cross_val_test(diff_AB=_diff(), test="sharp", fall_back_rho=0.1, mode="bogus")


def test_single_repetition_sharp_scheme_raises():
    X, y = _regression()
    with pytest.raises(ValueError, match="at least 2 repetitions"):
        sharp_cross_val_test(
            Ridge(0.1),
            Ridge(10.0),
            X,
            y,
            cv=RepeatedKFold(n_splits=5, n_repeats=1),
            random_state=0,
        )


def test_mismatched_X_y_raises():
    X, y = _regression()
    with pytest.raises(ValueError, match="same number of rows"):
        sharp_cross_val_test(Ridge(), Ridge(), X, y[:-1], cv=RepeatedKFold(n_splits=5, n_repeats=3))
