# Record a Refinement Round

Appends one row per pass through discovery, coding, and revision. The
round number increments on its own, so repeated calls build the record
in order.

## Usage

``` r
log_round(
  rounds = NULL,
  n_categories,
  n_coded,
  n_uncoded,
  notes = NA_character_
)
```

## Arguments

- rounds:

  Existing round table, or `NULL` to start one.

- n_categories:

  Categories in the codebook for this round.

- n_coded:

  Units that received at least one code.

- n_uncoded:

  Units that received none, from
  [`uncoded_units()`](https://mshin77.github.io/TextAnalysisR/reference/uncoded_units.md).

- notes:

  What changed this round. Optional.

## Value

The round table with one row appended.

## See also

[`round_summary()`](https://mshin77.github.io/TextAnalysisR/reference/round_summary.md)
for the coverage trend across rounds;
[`uncoded_units()`](https://mshin77.github.io/TextAnalysisR/reference/uncoded_units.md)
for the count this records.
