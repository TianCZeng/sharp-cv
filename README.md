# sharp-cv

Statistical tests for comparing predictive performance with re-designed
cross-validation, built on scikit-learn.

Predictive performance estimates from cross-validation folds are not
independent, so a paired t-test assuming independent estimates cannot
control the false positive rate. `sharp-cv` implements the **SHARP**
(Split-Half Analysis of Repeated Performance) test, which on every
repetition splits the data into two disjoint halves and runs a
cross-validation inside each half. Because the halves share no
observations, the two results of one repetition are independent, while
results from different repetitions remain correlated. That structure lets
the variance of a split-half result and the across-repetition correlation
both be estimated from the data instead of assumed.

`sharp-cv` also provides the **SHA** (Split-HAlf) test, the single-run
variant that splits the data once and keeps the individual folds. SHA is
cheaper, but it is described only as a variant in the paper and was not
benchmarked yet, so its false positive rate and power have not been
characterised the way SHARP's have. Prefer SHARP whenever the compute
budget allows repetition; see [How many repetitions?](#how-many-repetitions)
for suggested counts.

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

Regression, repeated 5-fold with 30 repetitions:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sharp_cv import sharp_cross_val_test

X, y = make_regression(n_samples=200, noise=1.0, random_state=0)
result = sharp_cross_val_test(
    Ridge(alpha=0.1), Ridge(alpha=10.0),
    X, y,
    cv=RepeatedKFold(n_splits=5, n_repeats=30),
    random_state=0,
)
print(result.statistic, result.pvalue)   # z and two-sided p
print(result.test, result.diff_AB.shape)  # 'sharp', (30, 2)
```

Classification is stratified automatically when `estimator_1` is a
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
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=30),
    scoring="accuracy",
    random_state=0,
)
```

Monte-Carlo cross-validation, 150 repetitions with a 20% test set inside
each half (`n_splits` counts repetitions here, not inner folds):

```python
from sklearn.model_selection import ShuffleSplit

result = sharp_cross_val_test(
    Ridge(alpha=0.1), Ridge(alpha=10.0), X, y,
    cv=ShuffleSplit(n_splits=150, test_size=0.2),
    random_state=0,
)
```

A single 5-fold run with no repetition uses the SHA test. This is also
what the default `cv=5` does, so pass a repeated splitter when you want
SHARP:

```python
result = sharp_cross_val_test(Ridge(alpha=0.1), Ridge(alpha=10.0), X, y,
                              cv=5, random_state=0)
print(result.test)   # 'sha'
```

## The procedure

One repetition of the split-half procedure is:

1. Divide the data at random into two disjoint halves A and B (stratified
   by class for classifiers).
2. Inside each half, run the inner cross-validation. Both estimators are
   fit on the same training folds and scored on the same test folds, which
   gives `score_1` and `score_2`.
3. Reduce each half to one mean difference, the mean over
   `score_1 - score_2`.

Repeating this J times gives a `[J, 2]` array of paired differences.
Values from the same repetition come from disjoint data and are
uncorrelated; values from different repetitions share data and are
correlated. SHARP estimates that correlation from the two halves and uses
it to compute the variance of the overall mean difference.

Throughout the package, **A and B always mean the two halves of the data,
never the two models**. The two models are 1 and 2, the tested difference
is model 1 minus model 2, and the `AB` in `diff_AB` marks its columns as
half A and half B.

The `cv` argument sets both the inner cross-validation and the number of
repetitions:

| `cv`                                               | Inside each half, per repetition  | Rows of `diff_AB` | Test  |
|----------------------------------------------------|-----------------------------------|-------------------|-------|
| `RepeatedKFold(n_splits=K, n_repeats=J)`           | K-fold, folds averaged            | J repetitions     | SHARP |
| `RepeatedStratifiedKFold(n_splits=K, n_repeats=J)` | stratified K-fold, folds averaged | J repetitions     | SHARP |
| `ShuffleSplit(n_splits=J, test_size=t)`            | one train/test split             | J repetitions     | SHARP |
| `StratifiedShuffleSplit(n_splits=J, test_size=t)`  | one stratified train/test split   | J repetitions     | SHARP |
| `KFold(K)`, `StratifiedKFold(K)` or an int `K`     | K-fold, folds kept                | K folds           | SHA   |

The halves are redrawn on every repetition. Running repeated
cross-validation inside two fixed halves would give a different
correlation structure and is not what SHARP assumes.

### How many repetitions?

A reasonable starting point is around 30 repetitions for repeated K-fold
and around 150 for Monte-Carlo. The stratified variants take the same
numbers. Note which argument carries the repetition count: `n_repeats`
for the repeated splitters, `n_splits` for the `ShuffleSplit` ones.

These are suggestions rather than requirements, but fewer repetitions
may decrease statistical power. Each repetition fits both estimators `2K`
times (or twice for Monte-Carlo schemes), so the cost is `2 * J * K` fits
per estimator, and repetition is the main cost of the procedure.

## Estimator modes

`mode` selects how the noise variance and the correlation are estimated:

| `mode` | Description                                                                |
|--------|----------------------------------------------------------------------------|
| `st`   | Score test: variance estimated under the null of zero difference. Default. |
| `lrt`  | Likelihood ratio test                                                      |
| `ml`   | Maximum likelihood, Wald z-test                                            |
| `rml`  | Restricted maximum likelihood, Wald z-test                                 |
| `mm`   | Method of moments, Wald z-test                                             |
| `mmc`  | Method of moments with the correlation clipped to `[0, rho_clip]`          |

In the simulations behind the method, the score test gave the best control
of the false-positive rate and is the default. Those simulations covered
SHARP only.

The four likelihood-based modes parameterise the correlation as
`tanh(r**2) * rho_max`, so they return a value in `[0, rho_max)` and never
a negative correlation. `rho_max` is the point at which that test's
covariance stops being positive definite, and it differs between the two
tests: `0.499` for SHARP, whose covariance is singular at `0.5`, and
`0.999` for SHA, whose block-diagonal covariance stays positive definite
up to `1`. `mmc` clips to `rho_clip`, which sits just inside `rho_max`
(`0.497` and `0.997`). Only `mm` leaves the correlation unconstrained.

`fall_back_rho` is used by `mm` and `mmc` only, and is ignored by every
other mode including the default: when the estimated variance of the mean
drops below the independent-samples minimum, the correlation is replaced
by this value. `sharp_cross_val_test` picks `1 / (2K)` for K-fold inner
schemes and `test_size / 2` for Monte-Carlo schemes. Both are the fraction
of the *full* dataset held out by one inner test set (the halving converts
a fraction of a half into a fraction of the whole), the heuristic used in
the paper's simulations. `sharp_test` and `sha_test` require it for `mm`
and `mmc` and accept `None` otherwise.

The fallback rule, the `mmc` mode and the bound on the correlation in the
likelihood-based modes are implementation safeguards. The paper describes
the five estimators (`st`, `lrt`, `ml`, `rml`, `mm`) without them; its
results used `st`, which the fallback rule never touches.

## Using your own paired differences

If you already ran the procedure yourself, pass the `[n, 2]` array
directly, with column 0 from half A and column 1 from half B. State how
the rows were produced, since it cannot be inferred from the array:

```python
from sharp_cv import sharp_test, sha_test, sharp_cross_val_test

# rows = repetitions of the split-half procedure, halves redrawn each time
z, p = sharp_test(diff_AB)                 # score test, the default

# rows = folds of a single K-fold run inside two fixed halves
z, p = sha_test(diff_AB)

# same, through the high-level entry point
result = sharp_cross_val_test(diff_AB=diff_AB, test="sharp")

# mm and mmc need fall_back_rho: 1 / (2K) for K-fold inside each half,
# test_size / 2 for a single Monte-Carlo split inside each half
z, p = sharp_test(diff_AB, fall_back_rho=1 / (2 * K), mode="mm")
```

Both functions return `(nan, nan)` when fewer than two rows are given or
the two columns are identical.

## Result object

`sharp_cross_val_test` returns a `SharpTestResult` named tuple with fields
`statistic`, `pvalue`, `test` (`'sharp'` or `'sha'`), `mode`,
`fall_back_rho` and `diff_AB`, the array that was tested. `fall_back_rho`
is the value that was in effect, or `None` when `diff_AB` was given
without one and the mode does not use it.

## Notes

- With `scoring=None` both estimators are scored with their own `score`
  method, so they must measure the same thing (for example both R² or
  both accuracy). Pass `scoring=` to be explicit.
- `random_state` seeds the half-splits and, for repeated and Monte-Carlo
  schemes, the inner splits too. Only `n_splits`, `n_repeats`, `test_size`
  and `train_size` are read from a `RepeatedKFold`,
  `RepeatedStratifiedKFold`, `ShuffleSplit` or `StratifiedShuffleSplit`;
  its own `random_state` is ignored, and a warning is raised when one is
  set. A plain `KFold` or `StratifiedKFold` object is used as given and
  keeps its own `shuffle` and `random_state`, so give it a `random_state`
  when it shuffles, or the result is not reproducible.
- There is no `groups` argument currently. The half-split is not group-aware, so
  samples from one subject or site can land in both halves, and group-aware
  splitters such as `GroupKFold` are rejected.
- `X` and `y` may be NumPy arrays, pandas objects or SciPy sparse matrices.
  pandas inputs are indexed with `.iloc`, so pipelines that select columns
  by name work.
- Estimators are cloned before every fit; pass unfitted estimators or
  pipelines.
- For hyperparameter tuning, pass a `GridSearchCV` (or a pipeline that
  contains one) as the estimator. It is refit inside every training fold,
  which gives the nested cross-validation the paper describes.
- Cross-validation runs inside each half rather than on the full dataset,
  so relative model performance may differ from full-dataset estimates,
  particularly when the algorithms or biomarkers being compared scale
  differently with training size.

## Citation

If you use this package, please cite:

> Zeng, T., Li, H., Zhang, S., Tan, Y. Q., Tian, F., Orban, C., ...
> Nichols, T. E., & Yeo, B. T. T. (2026). Spurious model comparisons are
> widespread in biomedical artificial intelligence. *bioRxiv*.
> https://doi.org/10.64898/2026.05.17.724301

The link resolves to the most recent version of the preprint.

## License

MIT. See [LICENSE](LICENSE).
