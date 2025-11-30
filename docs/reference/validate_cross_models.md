# Cross-Analysis Validation

Performs cross-validation between different analysis types (STM,
semantic, clustering).

## Usage

``` r
validate_cross_models(semantic_results, stm_results = NULL, verbose = TRUE)
```

## Arguments

- semantic_results:

  Results from semantic analysis.

- stm_results:

  Optional STM results for comparison.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing cross-validation metrics.
