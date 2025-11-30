# Plot Top Documents by Readability

Creates a bar plot of documents ranked by readability metric.

## Usage

``` r
plot_top_readability_documents(
  readability_data,
  metric,
  top_n = 15,
  order = "highest",
  title = NULL
)
```

## Arguments

- readability_data:

  Data frame from calculate_text_readability()

- metric:

  Metric to plot

- top_n:

  Number of documents to show (default: 15)

- order:

  Direction: "highest" or "lowest" (default: "highest")

- title:

  Plot title (default: auto-generated)

## Value

A plotly bar chart
