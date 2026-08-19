# Compare Human Codes With Machine Categories

Cross-tabulates human against model categories over the same documents
and reports the adjusted Rand index, which corrects for chance and does
not require the two schemes to share labels or count.

## Usage

``` r
align_categories(human, machine, unassigned = 0)
```

## Arguments

- human:

  Vector of human-assigned categories.

- machine:

  Vector of model-assigned categories, same length and order.

- unassigned:

  Label marking a document as unassigned in either vector, dropped
  before comparison. Defaults to `0`.

## Value

A list with `crosstab` (human against machine), `adjusted_rand`,
`best_match` (each human category paired with the machine category it
overlaps most, and the share of the human category that pairing covers),
and `n` (documents compared, excluding missing values).

## See also

[`validate_categories()`](https://mshin77.github.io/TextAnalysisR/reference/validate_categories.md)
for supervised confirmation.
