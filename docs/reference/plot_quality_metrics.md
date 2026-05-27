# Plot Topic Model Quality Metrics

Creates individual diagnostic metric plots across different K values
from stm::searchK results.

## Usage

``` r
plot_quality_metrics(search_results)
```

## Arguments

- search_results:

  Results from stm::searchK or find_optimal_k()

## Value

A named list of ggplot objects, one per available metric (possible keys:
semcoh, residual, heldout, lbound).
