# Plot Readability Distribution

Creates a boxplot showing the overall distribution of a readability
metric.

## Usage

``` r
plot_readability_distribution(readability_data, metric, title = NULL)
```

## Arguments

- readability_data:

  Data frame from calculate_text_readability()

- metric:

  Metric to plot (e.g., "flesch", "flesch_kincaid", "gunning_fog")

- title:

  Plot title (default: auto-generated)

## Value

A plotly boxplot

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c("Simple text.", "More complex sentence structure here.")
readability <- calculate_text_readability(texts)
plot <- plot_readability_distribution(readability, "flesch")
print(plot)
} # }
```
