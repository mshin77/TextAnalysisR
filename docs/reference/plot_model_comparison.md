# Plot Topic Model Comparison Scatter

Creates a scatter plot comparing topic model metrics across K values.
Automatically selects the best available metric combination.

## Usage

``` r
plot_model_comparison(
  search_results,
  title = "Coherence-Exclusivity Frontier (choose K in the upper-right)",
  height = 600,
  width = 800
)
```

## Arguments

- search_results:

  Results from stm::searchK or find_optimal_k()

- title:

  Plot title (default: "Model Comparison")

- height:

  Plot height in pixels (default: 600)

- width:

  Plot width in pixels (default: 800)

## Value

A plotly scatter plot
