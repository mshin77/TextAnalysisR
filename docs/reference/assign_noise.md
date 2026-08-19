# Assign Unclustered Documents to Confirmed Categories

Places documents left unassigned by density-based clustering, by
similarity to confirmed category members. Distance is returned so weak
placements stay visible, and `max_distance` leaves distant documents
unassigned.

## Usage

``` r
assign_noise(
  embeddings,
  categories,
  k = 5,
  max_distance = NULL,
  unassigned = 0
)
```

## Arguments

- embeddings:

  Numeric matrix or data frame, one row per document.

- categories:

  Category labels; `NA` or any value in `unassigned` marks a document as
  not yet placed.

- k:

  Neighbours to consult (default 5).

- max_distance:

  Cosine distance beyond which a document stays unassigned. `NULL`
  (default) places every document.

- unassigned:

  Label marking a document as not yet placed. Defaults to `0`, which
  [`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md)
  emits for outliers.

## Value

A tibble with `index`, `assigned`, and `distance`, one row per
previously unassigned document.

## See also

[`validate_categories()`](https://mshin77.github.io/TextAnalysisR/reference/validate_categories.md)
to test the categories afterwards.
