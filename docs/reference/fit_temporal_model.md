# Fit Temporal Topic Model

Analyzes how topics evolve over time by fitting topic models to
different time periods and tracking semantic changes.

## Usage

``` r
fit_temporal_model(
  texts,
  dates,
  time_windows = "yearly",
  embeddings = NULL,
  verbose = TRUE
)
```

## Arguments

- texts:

  A character vector of text documents to analyze.

- dates:

  A vector of dates corresponding to each document (will be converted to
  Date).

- time_windows:

  Time grouping strategy: "yearly", "monthly", or "quarterly" (default:
  "yearly").

- embeddings:

  Optional pre-computed embeddings matrix. If NULL, embeddings will be
  generated.

- verbose:

  Logical indicating whether to print progress messages (default: TRUE).

## Value

A list containing temporal analysis results with topic evolution
patterns.
