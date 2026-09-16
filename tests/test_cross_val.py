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

from sharp_cv import (
    SharpTestResult,
    sharp_confint,
    sharp_cross_val_test,
    sharp_test,
)
from sharp_cv.cross_val import _build_diff_AB, _resolve_scheme

MAX_SEED = 2**31 - 1


def _check_result(r: SharpTestResult, test: str, shape) -> None:
    assert isinstance(r, SharpTestResult)
    assert r.test == test
    assert r.diff_AB.shape == shape
    assert r.score_1_AB.shape == shape
    assert r.score_2_AB.shape == shape
    np.testing.assert_allclose(r.score_1_AB - r.score_2_AB, r.diff_AB, atol=1e-12)
    assert r.mean_diff == pytest.approx(r.diff_AB.mean())
    assert np.isfinite(r.statistic)
    assert 0.0 <= r.pvalue <= 1.0
    assert r.confidence_level == 0.95
    assert r.ci_low < r.mean_diff < r.ci_high
    assert (r.ci_low <= 0.0 <= r.ci_high) == (r.pvalue >= 0.05)


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


@pytest.mark.parametrize(
    "cv", [5, KFold(4), KFold(3, shuffle=True), StratifiedKFold(5)]
)
def test_single_run_splitters_are_rejected(cv):
    X, y = _regression()
    with pytest.raises(
        NotImplementedError, match="no repetition.*Pass a repeated splitter"
    ):
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
    proc = _build_diff_AB(
        Ridge(0.1),
        Ridge(10.0),
        X,
        y,
        cv=ShuffleSplit(n_splits=6, test_size=test_size),
        random_state=0,
    )
    assert proc.test == "sharp"
    assert proc.diff_AB.shape == (6, 2)
    assert proc.fall_back_rho == pytest.approx(expected_rho)


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
    proc = _build_diff_AB(
        _Recorder(),
        _Recorder(),
        X,
        y,
        cv=ShuffleSplit(n_splits=J, test_size=0.2),
        random_state=0,
    )
    assert proc.test == "sharp"
    assert proc.diff_AB.shape == (J, 2)
    assert proc.fall_back_rho == pytest.approx(0.1)
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


_SEEDED_SPLITTERS = [
    lambda rs: RepeatedKFold(n_splits=3, n_repeats=3, random_state=rs),
    lambda rs: ShuffleSplit(n_splits=3, test_size=0.2, random_state=rs),
]


@pytest.mark.parametrize("make_cv", _SEEDED_SPLITTERS)
def test_function_seed_wins_over_splitter_seed_with_warning(make_cv):
    X, y = _regression()
    results = []
    for rs in (0, 999):
        with pytest.warns(UserWarning, match="Both the .* were given a random_state"):
            results.append(
                sharp_cross_val_test(
                    Ridge(0.1), Ridge(10.0), X, y, cv=make_cv(rs), random_state=0
                )
            )
    np.testing.assert_array_equal(results[0].diff_AB, results[1].diff_AB)


@pytest.mark.parametrize("make_cv", _SEEDED_SPLITTERS)
def test_splitter_seed_alone_is_used_and_reproducible(make_cv):
    """Seeding only the splitter, the usual sklearn habit, is the same as
    seeding only sharp_cross_val_test with that value, and does not warn."""
    X, y = _regression()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r1 = sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, cv=make_cv(42))
        r2 = sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, cv=make_cv(42))
        r3 = sharp_cross_val_test(
            Ridge(0.1), Ridge(10.0), X, y, cv=make_cv(None), random_state=42
        )
        r4 = sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, cv=make_cv(7))
    np.testing.assert_array_equal(r1.diff_AB, r2.diff_AB)
    np.testing.assert_array_equal(r1.diff_AB, r3.diff_AB)
    assert not np.array_equal(r1.diff_AB, r4.diff_AB)


def test_unseeded_splitter_with_function_seed_does_not_warn():
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
        sharp_cross_val_test(
            Ridge(0.1), Ridge(10.0), X, y, cv=GroupKFold(3), random_state=0
        )


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
# Result object
# ---------------------------------------------------------------------------


def _small_result(**kw):
    X, y = _regression()
    kw.setdefault("cv", RepeatedKFold(n_splits=3, n_repeats=4))
    kw.setdefault("random_state", 0)
    return sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, **kw)


def test_statistic_has_the_sign_of_mean_diff():
    r = _small_result()
    assert r.mean_diff != 0
    assert np.sign(r.statistic) == np.sign(r.mean_diff)
    # Swapping the estimators flips both and swaps the score arrays.
    X, y = _regression()
    r_swapped = sharp_cross_val_test(
        Ridge(10.0),
        Ridge(0.1),
        X,
        y,
        cv=RepeatedKFold(n_splits=3, n_repeats=4),
        random_state=0,
    )
    assert r_swapped.mean_diff == pytest.approx(-r.mean_diff)
    assert r_swapped.statistic == pytest.approx(-r.statistic)
    np.testing.assert_array_equal(r_swapped.score_1_AB, r.score_2_AB)
    np.testing.assert_array_equal(r_swapped.score_2_AB, r.score_1_AB)


def test_scores_of_each_model_are_returned():
    X, y = _classification()
    r = sharp_cross_val_test(
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=10, random_state=0),
        X,
        y,
        cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=3),
        scoring="accuracy",
        random_state=0,
    )
    # Accuracies lie in [0, 1]; diff_AB is their difference.
    for scores in (r.score_1_AB, r.score_2_AB):
        assert scores.shape == (3, 2)
        assert np.all((0.0 <= scores) & (scores <= 1.0))
    np.testing.assert_allclose(r.score_1_AB - r.score_2_AB, r.diff_AB, atol=1e-12)
    assert r.mean_diff == pytest.approx(r.diff_AB.mean())


def test_repr_summarises_arrays():
    r = _small_result()
    text = repr(r)
    assert text.startswith("SharpTestResult(")
    assert "statistic=" in text and "mean_diff=" in text
    assert "ci_low=" in text and "ci_high=" in text
    assert "diff_AB=<array of shape (4, 2)>" in text
    assert "score_1_AB=<array of shape (4, 2)>" in text
    assert "score_2_AB=<array of shape (4, 2)>" in text
    assert "\n" not in text
    # The tuple interface is untouched.
    assert r[0] == r.statistic
    assert r._asdict()["diff_AB"] is r.diff_AB


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


def test_verbose_prints_progress(capsys):
    _small_result(verbose=10)
    captured = capsys.readouterr()
    assert "Parallel" in captured.out + captured.err


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_cv_is_required():
    X, y = _regression()
    with pytest.raises(TypeError, match="cv"):
        sharp_cross_val_test(Ridge(0.1), Ridge(10.0), X, y, random_state=0)


def test_diff_AB_and_test_kwargs_are_gone():
    """Precomputed arrays go to sharp_test; there is no test= to choose."""
    X, y = _regression()
    with pytest.raises(
        TypeError, match="unexpected keyword argument 'diff_AB'"
    ):
        sharp_cross_val_test(
            Ridge(0.1), Ridge(10.0), X, y, cv=5, diff_AB=np.zeros((8, 2))
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'test'"):
        sharp_cross_val_test(
            Ridge(0.1),
            Ridge(10.0),
            X,
            y,
            cv=RepeatedKFold(n_splits=5, n_repeats=3),
            test="sharp",
        )


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


def test_invalid_mode_raises_before_any_fitting():
    X, y = _regression()
    _Recorder.fits = []
    with pytest.raises(ValueError, match="mode must be one of"):
        sharp_cross_val_test(
            _Recorder(),
            _Recorder(),
            X,
            y,
            cv=RepeatedKFold(n_splits=5, n_repeats=3),
            mode="bogus",
        )
    assert _Recorder.fits == []


def test_mixed_classifier_and_regressor_rejected_without_scoring():
    X, y = _classification()
    _Recorder.fits = []
    with pytest.raises(
        ValueError, match="estimator_1 is a regressor and estimator_2 is a classifier"
    ):
        sharp_cross_val_test(
            _Recorder(),
            LogisticRegression(max_iter=1000),
            X,
            y,
            cv=RepeatedKFold(n_splits=3, n_repeats=2),
            random_state=0,
        )
    assert _Recorder.fits == []
    # An explicit scorer applies one metric to both and is allowed.
    r = sharp_cross_val_test(
        Ridge(0.1),
        LogisticRegression(max_iter=1000),
        X,
        y,
        cv=RepeatedKFold(n_splits=3, n_repeats=2),
        scoring="neg_mean_squared_error",
        random_state=0,
    )
    _check_result(r, "sharp", (2, 2))


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
        sharp_cross_val_test(
            Ridge(), Ridge(), X, y[:-1], cv=RepeatedKFold(n_splits=5, n_repeats=3)
        )


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------


def test_interval_matches_sharp_confint_on_the_returned_differences():
    """The driver must not invent its own interval: it hands the same
    ``diff_AB`` to the same inversion the standalone function uses."""
    for mode in ("st", "lrt", "mm", "rml"):
        r = _small_result(mode=mode, fall_back_rho=0.1)
        lo, hi = sharp_confint(r.diff_AB, 0.95, fall_back_rho=0.1, mode=mode)
        assert r.ci_low == pytest.approx(lo, rel=1e-12)
        assert r.ci_high == pytest.approx(hi, rel=1e-12)


@pytest.mark.parametrize("level", [0.80, 0.90, 0.99])
def test_confidence_level_is_honoured(level):
    r = _small_result(confidence_level=level)
    assert r.confidence_level == level
    lo, hi = sharp_confint(r.diff_AB, level)
    assert (r.ci_low, r.ci_high) == pytest.approx((lo, hi), rel=1e-12)
    wider = _small_result(confidence_level=0.999)
    assert (wider.ci_high - wider.ci_low) >= (r.ci_high - r.ci_low)


def test_confidence_level_none_skips_the_interval():
    r = _small_result(confidence_level=None)
    assert np.isnan(r.confidence_level)
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)
    # everything else is unchanged
    base = _small_result()
    assert r.statistic == pytest.approx(base.statistic)
    assert r.pvalue == pytest.approx(base.pvalue)


def test_result_is_still_a_named_tuple_with_the_old_fields_in_place():
    """``ci_*`` were appended, so existing positional access still works."""
    r = _small_result()
    assert r[0] == r.statistic
    assert r[1] == r.pvalue
    assert r[2] == r.mean_diff
    assert SharpTestResult._fields[:9] == (
        "statistic",
        "pvalue",
        "mean_diff",
        "test",
        "mode",
        "fall_back_rho",
        "diff_AB",
        "score_1_AB",
        "score_2_AB",
    )
