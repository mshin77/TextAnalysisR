# Retrieve Analytic Memos

Returns memos filtered by target and round. Omitted filters match
everything, so calling with no filter returns the full table in the
order written.

## Usage

``` r
get_memos(memos, target_type = NULL, target_id = NULL, round = NULL)
```

## Arguments

- memos:

  Memo table from
  [`add_memo()`](https://mshin77.github.io/TextAnalysisR/reference/add_memo.md).

- target_type:

  "unit", "category", or `NULL` for both.

- target_id:

  Identifier to match, or `NULL` for all.

- round:

  Round to match, or `NULL` for all.

## Value

A tibble of matching memos, oldest first.

## See also

[`add_memo()`](https://mshin77.github.io/TextAnalysisR/reference/add_memo.md)
to write them.
