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
    train_test_split,
)

from sharp_cv import (
    SharpTestResult,
    sharp_confint,
    sharp_cross_val_test,
    sharp_test,
)
from sharp_cv.cross_val import _build_diff_AB

MAX_SEED = 2**31 - 1


def _check_result(r: SharpTestResult, shape) -> None:
    assert isinstance(r, SharpTestResult)
    assert r.test == "sharp"
    assert r.diff_AB.shape == r.score_1_AB.shape == r.score_2_AB.shape == shape
    np.testing.assert_allclose(r.score_1_AB - r.score_2_AB, r.diff_AB, atol=1e-12)
    assert r.mean_diff == pytest.approx(r.diff_AB.mean())
    assert np.isfinite(r.statistic)
    assert r.ci_low < r.mean_diff < r.ci_high


def _regression():
    return make_regression(n_samples=200, noise=1.0, random_state=0)


def _classification():
    return make_classification(
        n_samples=400, n_features=10, n_informative=5, random_state=0
    )


def _small_result(**kw):
    X, y = _regression()
    kw.setdefault("cv", RepeatedKFold(n_splits=3, n_repeats=4))
    kw.setdefault("random_state", 0)
    return sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, **kw)


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


def test_bad_arguments_raise_before_any_fitting():
    X, y = _regression()
    cv = RepeatedKFold(n_splits=3, n_repeats=2)
    cases = [
        (dict(cv=5), NotImplementedError, "cv=KFold runs a single"),
        (dict(cv=GroupKFold(3)), ValueError, "GroupKFold is not supported"),
        (
            dict(cv=RepeatedKFold(n_splits=3, n_repeats=1)),
            ValueError,
            "at least 2 repetitions",
        ),
        (dict(cv=cv, mode="all"), ValueError, "mode='all' was removed"),
        (dict(cv=cv, mode="bogus"), ValueError, "mode must be one of"),
        (dict(cv=cv, confidence_level=1.0), ValueError, "level must lie in"),
        (dict(cv=cv, fall_back_rho=0.5), ValueError, "fall_back_rho must lie in"),
        (
            dict(cv=cv, mode="mm", fall_back_rho=-0.1),
            ValueError,
            "fall_back_rho must lie in",
        ),
    ]
    _Recorder.fits = []
    for kw, error, match in cases:
        with pytest.raises(error, match=match):
            sharp_cross_val_test(_Recorder(), _Recorder(), X, y, **kw)
    with pytest.raises(ValueError, match="same number of rows"):
        sharp_cross_val_test(_Recorder(), _Recorder(), X, y[:-1], cv=cv)
    with pytest.raises(ValueError, match="estimator_1 is a regressor"):
        sharp_cross_val_test(_Recorder(), LogisticRegression(), X, y, cv=cv)
    assert _Recorder.fits == []


@pytest.mark.parametrize(
    "n, cv, match",
    [
        # Halves of 4 and 5 cannot hold 5 folds.
        (9, RepeatedKFold(n_splits=5, n_repeats=2), "halves of 4 and 5"),
        # 5 folds in halves of 9 and 10 leave a test fold of 1 sample.
        (19, RepeatedKFold(n_splits=5, n_repeats=2), "inner test sets of 1 sample"),
        (10, ShuffleSplit(n_splits=2, test_size=0.2), "inner test sets of 1 sample"),
    ],
)
def test_data_too_small_for_cv_raise_before_any_fitting(n, cv, match):
    X, y = _regression()
    _Recorder.fits = []
    with pytest.raises(ValueError, match=match):
        sharp_cross_val_test(_Recorder(), _Recorder(), X[:n], y[:n], cv=cv)
    assert _Recorder.fits == []
    # 20 samples are enough for each of these: 4 * K for K = 5, and a test
    # set of 2 from a half of 10 at test_size=0.2.
    r = sharp_cross_val_test(
        Ridge(0.1), Ridge(10.0), X[:20], y[:20], cv=cv, random_state=0
    )
    assert np.isfinite(r.statistic)


def test_non_finite_scores_warn_in_the_calling_process():
    """sklearn's own warning is raised in the worker and lost with n_jobs."""

    def nan_scorer(estimator, X_test, y_test):
        return float("nan")

    with pytest.warns(UserWarning, match="2 of 2 rows .* not finite"):
        r = _small_result(
            cv=RepeatedKFold(n_splits=3, n_repeats=2), scoring=nan_scorer, n_jobs=2
        )
    assert np.isnan(r.statistic) and np.isnan(r.pvalue)
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)


# ---------------------------------------------------------------------------
# The procedure itself
# ---------------------------------------------------------------------------


def test_repeated_kfold_rows_are_fold_means_of_fresh_halves():
    """Reproduce the procedure by hand, including the random stream."""
    X, y = _regression()
    K, J = 5, 3
    # Both seeds are given, so the splitter's is ignored (with a warning)
    # and only the random_state passed to the procedure drives the streams
    # below.
    with pytest.warns(UserWarning, match="Both the RepeatedKFold"):
        proc = _build_diff_AB(
            Ridge(0.1),
            Ridge(10.0),
            X,
            y,
            cv=RepeatedKFold(n_splits=K, n_repeats=J, random_state=123),
            random_state=0,
        )

    rng = np.random.RandomState(0)
    scorer = check_scoring(Ridge(0.1), scoring=None)
    expected, expected_1, expected_2 = [], [], []
    for _ in range(J):
        X_A, X_B, y_A, y_B = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=rng.randint(MAX_SEED),
        )
        inner = KFold(n_splits=K, shuffle=True, random_state=rng.randint(MAX_SEED))
        row, row_1, row_2 = [], [], []
        for X_h, y_h in ((X_A, y_A), (X_B, y_B)):
            d, s1, s2 = [], [], []
            for tr, te in inner.split(X_h, y_h):
                a = clone(Ridge(0.1)).fit(X_h[tr], y_h[tr])
                b = clone(Ridge(10.0)).fit(X_h[tr], y_h[tr])
                s1.append(scorer(a, X_h[te], y_h[te]))
                s2.append(scorer(b, X_h[te], y_h[te]))
                d.append(s1[-1] - s2[-1])
            row.append(np.mean(d))
            row_1.append(np.mean(s1))
            row_2.append(np.mean(s2))
        expected.append(row)
        expected_1.append(row_1)
        expected_2.append(row_2)
    np.testing.assert_allclose(proc.diff_AB, np.asarray(expected), rtol=1e-12)
    np.testing.assert_allclose(proc.score_1_AB, np.asarray(expected_1), rtol=1e-12)
    np.testing.assert_allclose(proc.score_2_AB, np.asarray(expected_2), rtol=1e-12)
    assert proc.fall_back_rho == pytest.approx(1 / (2 * K))


def test_monte_carlo_uses_one_split_per_half_per_repetition():
    n, J = 60, 5
    rng = np.random.default_rng(1)
    X = np.column_stack([np.arange(n), rng.normal(size=n)])
    y = 2.0 * X[:, 1] + rng.normal(size=n)

    _Recorder.fits = []
    proc = _build_diff_AB(
        _Recorder(),
        _Recorder(),
        X,
        y,
        cv=ShuffleSplit(n_splits=J, test_size=0.2),
        random_state=0,
    )
    assert proc.diff_AB.shape == (J, 2)
    fits = _Recorder.fits
    assert len(fits) == J * 2 * 2  # repetitions x halves x models
    # Training set is 80% of a 30-sample half.
    assert all(len(s) == 24 for s in fits)
    # Both models of a half are fit on the same rows; the halves of one
    # repetition are disjoint; the halves change across repetitions.
    assert all(fits[2 * i] == fits[2 * i + 1] for i in range(2 * J))
    half_A_train = [fits[4 * j] for j in range(J)]
    half_B_train = [fits[4 * j + 2] for j in range(J)]
    assert all(a.isdisjoint(b) for a, b in zip(half_A_train, half_B_train))
    assert len(set(half_A_train)) == J


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
    proc = _build_diff_AB(
        Ridge(0.1),
        Ridge(10.0),
        X,
        y,
        cv=ShuffleSplit(n_splits=3, test_size=test_size),
        random_state=0,
    )
    assert proc.fall_back_rho == pytest.approx(expected_rho)


@pytest.mark.parametrize(
    "make_cv",
    [
        lambda rs: RepeatedKFold(n_splits=3, n_repeats=3, random_state=rs),
        lambda rs: ShuffleSplit(n_splits=3, test_size=0.2, random_state=rs),
    ],
)
def test_seed_comes_from_the_function_or_the_splitter(make_cv):
    """Seeding only the splitter, the usual sklearn habit, is the same as
    seeding only sharp_cross_val_test with that value, and does not warn.
    When both are seeded the function's seed wins, with a warning."""
    X, y = _regression()

    def run(cv, **kw):
        return sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, cv=cv, **kw).diff_AB

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        splitter_only = run(make_cv(42))
        function_only = run(make_cv(None), random_state=42)
        other_seed = run(make_cv(7))
    np.testing.assert_array_equal(splitter_only, function_only)
    assert not np.array_equal(splitter_only, other_seed)
    with pytest.warns(UserWarning, match="Both the .* were given a random_state"):
        both = run(make_cv(7), random_state=42)
    np.testing.assert_array_equal(both, function_only)


def test_n_jobs_does_not_change_the_result():
    r_seq = _small_result()
    r_par = _small_result(n_jobs=2)
    # Same repetitions in the same order, so the only difference is the BLAS
    # thread count, which joblib pins to 1 inside its worker processes.
    tol = dict(rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(r_seq.diff_AB, r_par.diff_AB, **tol)
    np.testing.assert_allclose(r_seq.score_1_AB, r_par.score_1_AB, **tol)
    np.testing.assert_allclose(r_seq.score_2_AB, r_par.score_2_AB, **tol)
    assert r_seq.statistic == pytest.approx(r_par.statistic)


# ---------------------------------------------------------------------------
# Inputs and result
# ---------------------------------------------------------------------------


def test_classifiers_with_a_scoring_string():
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
    _check_result(r, (3, 2))
    assert r.fall_back_rho == pytest.approx(1 / 10)
    for scores in (r.score_1_AB, r.score_2_AB):
        assert np.all((0.0 <= scores) & (scores <= 1.0))


def test_result_matches_sharp_test_and_sharp_confint():
    """The driver hands the returned ``diff_AB`` and ``fall_back_rho`` (its
    default, or the one given) to the same functions a user would call, at
    the requested level."""
    for mode, given in (("st", None), ("mm", 0.2)):
        r = _small_result(mode=mode, fall_back_rho=given, confidence_level=0.9)
        rho = r.fall_back_rho
        assert given is None or rho == given
        z, p = sharp_test(r.diff_AB, fall_back_rho=rho, mode=mode)
        lo, hi = sharp_confint(r.diff_AB, 0.9, fall_back_rho=rho, mode=mode)
        assert (r.statistic, r.pvalue) == pytest.approx((z, p), rel=1e-12)
        assert (r.ci_low, r.ci_high) == pytest.approx((lo, hi), rel=1e-12)
    r = _small_result(confidence_level=None)
    assert np.isnan(r.confidence_level)
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)


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
    # Not bit-for-bit: a DataFrame of one float block reaches BLAS as an
    # F-contiguous array and a plain ndarray as a C-contiguous one, which
    # moves the last bits of the fitted coefficients.
    np.testing.assert_allclose(r_np.diff_AB, r_df.diff_AB, rtol=1e-9, atol=1e-12)

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
    _check_result(r, (2, 2))
