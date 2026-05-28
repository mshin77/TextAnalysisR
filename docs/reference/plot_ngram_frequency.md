# Plot N-gram Frequency

Creates a bar plot showing n-gram frequencies with optional highlighting
of selected n-grams. Supports both detected n-grams and selected
multi-word expressions.

## Usage

``` r
plot_ngram_frequency(
  ngram_data,
  top_n = 30,
  selected = NULL,
  title = "N-gram Frequency",
  highlight_color = "#10B981",
  default_color = "#6B7280",
  height = 500,
  width = NULL,
  show_stats = TRUE
)
```

## Arguments

- ngram_data:

  Data frame containing n-gram data with columns:

  - `collocation`: The n-gram text

  - `count`: Frequency count

  - `lambda`: (optional) Lambda statistic

  - `z`: (optional) Z-score statistic

- top_n:

  Number of top n-grams to display (default: 30)

- selected:

  Character vector of selected n-grams to highlight (default: NULL)

- title:

  Plot title (default: "N-gram Frequency")

- highlight_color:

  Color for highlighted bars (default: "#10B981")

- default_color:

  Color for non-highlighted bars (default: "#6B7280")

- height:

  Plot height in pixels (default: 500)

- width:

  Plot width in pixels (default: NULL for auto)

- show_stats:

  Whether to show lambda and z-score in hover (default: TRUE)

## Value

A plotly object

## Examples

``` r
# \donttest{
  ngram_df <- data.frame(
    collocation = c("machine learning", "deep learning", "neural network"),
    count = c(150, 120, 90),
    lambda = c(5.2, 4.8, 4.1),
    z = c(12.3, 10.5, 9.2)
  )
  plot_ngram_frequency(ngram_df, selected = c("machine learning"))

# }
```
