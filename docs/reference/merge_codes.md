# Combine Coder Assignment Files

Reads per-coder assignment files (each a saved assignments tibble) and
binds them into one long assignments table for
[`code_agreement()`](https://mshin77.github.io/TextAnalysisR/reference/code_agreement.md).
Supports the async workflow where each coder codes a copy and exports
it.

## Usage

``` r
merge_codes(files)
```

## Arguments

- files:

  Character vector of `.rds` paths, or a list of data frames.

## Value

A tibble of combined, de-duplicated assignments.

## See also

[`code_agreement()`](https://mshin77.github.io/TextAnalysisR/reference/code_agreement.md)
for the agreement summary.
