# sharp-cv

Statistical tests for comparing two predictive models with cross-validation,
built on scikit-learn.

Cross-validation folds are not independent, so a paired t-test over fold
scores rejects far too often. `sharp-cv` implements the **SHARP** test,
which splits the data into two halves, runs cross-validation inside each
half, and estimates the fold correlation from the two halves so that the
variance of the mean difference is correct. It also provides **SHA**, the
single-run variant with no repetition.

## Install

```bash
pip install sharp-cv
```

or the development version:

```bash
pip install git+https://github.com/TianCZeng/sharp-cv.git
```

Requires Python 3.9 or later, NumPy, SciPy and scikit-learn.

## Quick start

Regression, repeated 5-fold with 20 repetitions:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sharp_cv import sharp_cross_val_test

X, y = make_regression(n_samples=200, noise=1.0, random_state=0)
result = sharp_cross_val_test(
    Ridge(alpha=0.1), Ridge(alpha=10.0),
    X, y,
    cv=RepeatedKFold(n_splits=5, n_repeats=20),
    random_state=0,
)
print(result.statistic, result.pvalue)   # z and two-sided p
print(result.test, result.diff_AB.shape)  # 'sharp', (20, 2)
```

Classification is stratified automatically when `estimator_a` is a
classifier:

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

X, y = make_classification(n_samples=400, n_classes=3, n_informative=5,
                           random_state=0)
result = sharp_cross_val_test(
    LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=50),
    X, y,
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=20),
    scoring="accuracy",
    random_state=0,
)
```

Monte-Carlo cross-validation, 100 repetitions with a 20% test set inside
each half:

```python
from sklearn.model_selection import ShuffleSplit

result = sharp_cross_val_test(
    Ridge(alpha=0.1), Ridge(alpha=10.0), X, y,
    cv=ShuffleSplit(n_splits=100, test_size=0.2),
    random_state=0,
)
```

A single 5-fold run with no repetition uses the SHA test:

```python
result = sharp_cross_val_test(Ridge(alpha=0.1), Ridge(alpha=10.0), X, y,
                              cv=5, random_state=0)
print(result.test)   # 'sha'
```

## The procedure

One repetition of the split-half procedure is:

1. Divide the data at random into two disjoint halves A and B
   (stratified by class for classifiers).
2. Inside each half, run the inner cross-validation. Both estimators are
   fit on the same training folds and scored on the same test folds.
3. Reduce each half to one number, the mean over the inner folds of
   `score_A - score_B`.

Repeating this J times gives a `[J, 2]` array of paired differences.
Values from the same repetition come from disjoint data and are
uncorrelated; values from different repetitions share data and are
correlated. SHARP estimates that correlation from the two halves and uses
it to compute the variance of the overall mean difference.

The `cv` argument sets both the inner cross-validation and the number of
repetitions:

| `cv`                                                   | Inside each half, per repetition | Rows of `diff_AB` | Test  |
|--------------------------------------------------------|----------------------------------|-------------------|-------|
| `RepeatedKFold(n_splits=K, n_repeats=J)`               | K-fold, folds averaged           | J repetitions     | SHARP |
| `RepeatedStratifiedKFold(n_splits=K, n_repeats=J)`     | stratified K-fold, folds averaged| J repetitions     | SHARP |
| `ShuffleSplit(n_splits=J, test_size=t)`                | one train/test split             | J repetitions     | SHARP |
| `StratifiedShuffleSplit(n_splits=J, test_size=t)`      | one stratified train/test split  | J repetitions     | SHARP |
| `KFold(K)`, `StratifiedKFold(K)` or an int `K`         | K-fold, folds kept               | K folds           | SHA   |

The halves are redrawn on every repetition. Running repeated
cross-validation inside two fixed halves would give a different
correlation structure and is not what SHARP assumes.

Each repetition fits both estimators `2K` times (or twice for Monte-Carlo
schemes), so the cost is `2 * J * K` fits per estimator.

## Estimator modes

`mode` selects how the noise variance and the fold correlation are
estimated:

| `mode` | Description                                                        |
|--------|--------------------------------------------------------------------|
| `st`   | Score test: variance estimated under the null of zero difference. Default. |
| `lrt`  | Likelihood ratio test                                              |
| `ml`   | Maximum likelihood, Wald z-test                                    |
| `rml`  | Restricted maximum likelihood, Wald z-test                         |
| `mm`   | Method of moments, Wald z-test                                     |
| `mmc`  | Method of moments with the correlation clipped to `[0, 0.497]`     |

In the simulations behind the method, the score test gave the best control
of the false-positive rate and is the default.

`fall_back_rho` only matters for `mm` and `mmc`: when the estimated
variance of the mean drops below the independent-samples minimum, the
correlation is replaced by this value. `sharp_cross_val_test` picks
`1 / (2K)` for K-fold inner schemes and `test_size / 2` for Monte-Carlo
schemes.

## Using your own paired differences

If you already ran the procedure yourself, pass the `[n, 2]` array
directly. State how the rows were produced and give `fall_back_rho`,
since neither can be inferred from the array:

```python
from sharp_cv import sharp_test, sha_test, sharp_cross_val_test

# rows = repetitions of the split-half procedure, K-fold inside each half
z, p = sharp_test(diff_AB, fall_back_rho=1 / (2 * K), mode="st")

# rows = folds of a single K-fold run inside two fixed halves
z, p = sha_test(diff_AB, fall_back_rho=1 / (2 * K), mode="st")

# same, through the high-level entry point
result = sharp_cross_val_test(diff_AB=diff_AB, test="sharp",
                              fall_back_rho=1 / (2 * K))
```

Both functions return `(nan, nan)` when fewer than two rows are given or
the two columns are identical.

## Result object

`sharp_cross_val_test` returns a `SharpTestResult` named tuple with fields
`statistic`, `pvalue`, `test` (`'sharp'` or `'sha'`), `mode`,
`fall_back_rho` and `diff_AB`, the array that was tested.

## Notes

- With `scoring=None` both estimators are scored with their own `score`
  method, so they must measure the same thing (for example both R² or
  both accuracy). Pass `scoring=` to be explicit.
- `random_state` seeds the half-splits and, for repeated and Monte-Carlo
  schemes, the inner splits too. A plain `KFold` or `StratifiedKFold`
  object is used as given and keeps its own `shuffle` and `random_state`.
- Estimators are cloned before every fit; pass unfitted estimators or
  pipelines.

## Citation

If you use this package, please cite the SHARP paper. A reference will be
added here once the manuscript is published.

## License

MIT. See [LICENSE](LICENSE).
