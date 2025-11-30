# Plot Readability by Group

Creates grouped boxplots comparing readability across categories.

## Usage

``` r
plot_readability_by_group(readability_data, metric, group_var, title = NULL)
```

## Arguments

- readability_data:

  Data frame from calculate_text_readability()

- metric:

  Metric to plot

- group_var:

  Name of grouping variable column

- title:

  Plot title (default: auto-generated)

## Value

A plotly boxplot
