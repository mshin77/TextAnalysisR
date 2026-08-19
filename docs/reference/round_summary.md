# Coverage Trend Across Refinement Rounds

Reports what share of units the codebook left untouched in each round,
and how that share moved. A falling share is evidence the codebook is
still learning from the text; a flat share is evidence it has stopped.

## Usage

``` r
round_summary(rounds)
```

## Arguments

- rounds:

  Round table from
  [`log_round()`](https://mshin77.github.io/TextAnalysisR/reference/log_round.md).

## Value

A tibble with `round`, `n_categories`, `n_units`, `n_coded`,
`n_uncoded`, `pct_uncoded`, and `change` in percentage points against
the previous round (`NA` for the first).

## See also

[`log_round()`](https://mshin77.github.io/TextAnalysisR/reference/log_round.md)
to build the table.
