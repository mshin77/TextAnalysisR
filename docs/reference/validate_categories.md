# Confirm Categories With Supervised Learning

Tests whether categories are recoverable from the text representation
alone. A classifier maps document embeddings to category labels under
stratified cross-validation. Categories a classifier cannot recover are
candidates for merging or revision.

## Usage

``` r
validate_categories(
  embeddings,
  categories,
  method = c("knn", "multinom"),
  folds = 5,
  k = 5,
  balance = c("none", "downsample"),
  unassigned = 0,
  seed = 123
)
```

## Arguments

- embeddings:

  Numeric matrix or data frame, one row per document.

- categories:

  Category labels, one per row. `NA` and any value in `unassigned` are
  excluded; place them first with
  [`assign_noise()`](https://mshin77.github.io/TextAnalysisR/reference/assign_noise.md).

- method:

  "knn" (default) or "multinom".

- folds:

  Cross-validation folds (default 5), stratified by category.

- k:

  Neighbours for `method = "knn"` (default 5).

- balance:

  "none" (default) or "downsample" to equalize category sizes within
  each training fold.

- unassigned:

  Label marking a document as unassigned, excluded before fitting.
  Defaults to `0`, which
  [`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md)
  emits for outliers. Pass `-1` for output from tools that use that
  convention.

- seed:

  Random seed for fold assignment and downsampling.

## Value

A list with `overall` (accuracy, macro and weighted F1, counts),
`by_category` (support, precision, recall, F1), `confusion` (a table of
actual against predicted), and `predictions` (per-document actual and
predicted labels).

## See also

[`assign_noise()`](https://mshin77.github.io/TextAnalysisR/reference/assign_noise.md)
to place unassigned documents before confirming;
[`align_categories()`](https://mshin77.github.io/TextAnalysisR/reference/align_categories.md)
to compare human codes against machine clusters;
[`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md)
to produce the categories being tested.
