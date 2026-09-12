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

## Install

```bash
pip install sharp-cv
```

or the development version:

```bash
pip install git+https://github.com/TianCZeng/sharp-cv.git
```

Requires Python 3.9 or later, NumPy, SciPy, scikit-learn and joblib.

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
print(result.mean_diff)                  # mean score difference, model 1 minus model 2
print(result.score_1_AB.mean(), result.score_2_AB.mean())  # each model's mean score
```

The tested difference is always model 1 minus model 2, so a negative
statistic means the second estimator scored higher. `result.diff_AB` is
the `[30, 2]` array of paired differences the test was run on.

Classification is stratified automatically when `estimator_1` is a
classifier. Repetitions are independent of each other, so `n_jobs` runs
them in parallel as in `cross_val_score`, and `verbose` prints progress:

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
    n_jobs=-1,
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

`cv` has no default: the number of repetitions is a choice, so it has to be
passed. A single-run splitter — an int, `KFold` or `StratifiedKFold` — has
no repetition to work with and is rejected.

## The procedure

One repetition of the split-half procedure is:

1. Divide the data at random into two disjoint halves A and B (stratified
   by class for classifiers).
2. Inside each half, run the inner cross-validation. Both estimators are
   fit on the same training folds and scored on the same test folds, which
   gives `score_1` and `score_2`.
3. Reduce each half to one mean score per estimator and one mean
   difference, the mean over `score_1 - score_2`.

Repeating this J times gives a `[J, 2]` array of paired differences.
Values from the same repetition come from disjoint data and are
uncorrelated; values from different repetitions share data and are
correlated. SHARP estimates that correlation from the two halves and uses
it to compute the variance of the overall mean difference.

Throughout the package, **A and B always mean the two halves of the data,
never the two models**. The two models are 1 and 2, the tested difference
is model 1 minus model 2, and the `AB` in `diff_AB`, `score_1_AB` and
`score_2_AB` marks their columns as half A and half B.

The `cv` argument sets both the inner cross-validation and the number of
repetitions:

| `cv`                                               | Inside each half, per repetition  | Rows of `diff_AB` |
|----------------------------------------------------|-----------------------------------|-------------------|
| `RepeatedKFold(n_splits=K, n_repeats=J)`           | K-fold, folds averaged            | J repetitions     |
| `RepeatedStratifiedKFold(n_splits=K, n_repeats=J)` | stratified K-fold, folds averaged | J repetitions     |
| `ShuffleSplit(n_splits=J, test_size=t)`            | one train/test split              | J repetitions     |
| `StratifiedShuffleSplit(n_splits=J, test_size=t)`  | one stratified train/test split   | J repetitions     |
| `KFold(K)`, `StratifiedKFold(K)` or an int `K`     | single run, no repetition         | rejected          |

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
per estimator, and repetition is the main cost of the procedure. Pass
`n_jobs` to spread the repetitions over several processes.

## Estimator modes

`mode` selects how the noise variance and the correlation are estimated.
We recommend the default, `st`.

| `mode` | Description                                                                |
|--------|----------------------------------------------------------------------------|
| `st`   | Score test: variance estimated under the null of zero difference. Default. |
| `lrt`  | Likelihood ratio test                                                      |
| `ml`   | Maximum likelihood, Wald z-test                                            |
| `rml`  | Restricted maximum likelihood, Wald z-test                                 |
| `mm`   | Method of moments, Wald z-test                                             |
| `mmc`  | Method of moments with the correlation clipped to `[0, rho_clip]`          |

The four likelihood-based modes parameterise the correlation as
`tanh(r**2) * rho_max`, so they return a value in `[0, rho_max)` and never
a negative correlation. `rho_max` is `0.499` for SHARP, the point at which
its covariance stops being positive definite (it is singular at `0.5`).
`mmc` clips to `rho_clip`, just inside that bound (`0.497`). Only `mm`
leaves the correlation unconstrained.

`fall_back_rho` is used by `mm` and `mmc` only, and is ignored by every
other mode including the default: when the estimated variance of the mean
drops below the independent-samples minimum, the correlation is replaced
by this value. `sharp_cross_val_test` picks `1 / (2K)` for K-fold inner
schemes and `test_size / 2` for Monte-Carlo schemes. Both are the fraction
of the *full* dataset held out by one inner test set (the halving converts
a fraction of a half into a fraction of the whole), the heuristic used in
the paper's simulations. `sharp_test` requires it for `mm` and `mmc` and
accepts `None` otherwise.

The fallback rule, the `mmc` mode and the bound on the correlation in the
likelihood-based modes are implementation safeguards. The paper describes
the five estimators (`st`, `lrt`, `ml`, `rml`, `mm`) without them; its
results used `st`, which the fallback rule never touches.

## Using your own paired differences

If you already ran the procedure yourself, pass the `[n, 2]` array to
`sharp_test`, with column 0 from half A and column 1 from half B. Each row
must be one repetition of the split-half procedure, with the halves
redrawn every time.

```python
from sharp_cv import sharp_test

z, p = sharp_test(diff_AB)                 # score test, the default

# mm and mmc need fall_back_rho: 1 / (2K) for K-fold inside each half,
# test_size / 2 for a single Monte-Carlo split inside each half
z, p = sharp_test(diff_AB, fall_back_rho=1 / (2 * K), mode="mm")
```

`sharp_test` returns `(nan, nan)` when fewer than two rows are given or the
two columns are identical.

## Result object

`sharp_cross_val_test` returns a `SharpTestResult` named tuple:

| Field                      | Contents                                                                                   |
|----------------------------|--------------------------------------------------------------------------------------------|
| `statistic`                | z statistic; positive when model 1 scored higher, negative when model 2 did                |
| `pvalue`                   | two-sided p-value                                                                          |
| `mean_diff`                | mean of `diff_AB`, the estimated difference in the units of the score                      |
| `test`                     | `'sharp'`                                                                                  |
| `mode`                     | the estimator mode that was used                                                           |
| `fall_back_rho`            | the fallback correlation in effect; read by `mm` and `mmc` only                            |
| `diff_AB`                  | `[J, 2]` paired differences, one row per repetition, half A and half B                     |
| `score_1_AB`, `score_2_AB` | `[J, 2]` mean score of each model in half A and half B; `diff_AB` is their difference      |

Printing the result shows the scalar fields and the shapes of the arrays.

## Notes

- With `scoring=None` both estimators are scored with their own `score`
  method, so they must measure the same thing (for example both R² or
  both accuracy). A classifier and a regressor are rejected in that case.
  Pass `scoring=` to be explicit.
- `random_state` seeds one random stream that draws the half-splits and,
  for repeated and Monte-Carlo schemes, the inner splits too. Pass it to
  `sharp_cross_val_test`, or set it on the `RepeatedKFold`,
  `RepeatedStratifiedKFold`, `ShuffleSplit` or `StratifiedShuffleSplit`
  you pass as `cv`; either alone makes the result reproducible. If both
  are set, the one given to `sharp_cross_val_test` wins and a warning is
  raised. Only `n_splits`, `n_repeats`, `test_size`, `train_size` and
  `random_state` are read from those splitters; their `split` method is
  never called.
- `n_jobs` runs repetitions in parallel through joblib, as
  `cross_val_score` does, and `verbose` prints progress. The splits, the
  fits and their order do not depend on `n_jobs`: every seed is drawn
  before any repetition runs. Only the last bits can move, because joblib
  pins BLAS to one thread inside its workers.
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

## The single-split SHA test

The paper also describes **SHA** (Split-HAlf), a cheaper variant that
splits the data once and keeps the individual folds. It is not included in
this release: a single split gives only two half results however many
folds are used, which leaves the test too little to estimate its own
uncertainty from, and in practice it almost never rejects. This is why
single-run splitters are rejected by `sharp_cross_val_test`.

## Citation

If you use this package, please cite:

> Zeng, T., Li, H., Zhang, S., Tan, Y. Q., Tian, F., Orban, C., ...
> Nichols, T. E., & Yeo, B. T. T. (2026). Spurious model comparisons are
> widespread in biomedical artificial intelligence. *bioRxiv*.
> https://doi.org/10.64898/2026.05.17.724301

The link resolves to the most recent version of the preprint.

## License

MIT. See [LICENSE](LICENSE).
