# Plot Sentiment by Category

Creates a grouped or stacked bar plot showing sentiment distribution
across categories.

## Usage

``` r
plot_sentiment_by_category(
  sentiment_data,
  category_var,
  plot_type = "bar",
  title = NULL
)
```

## Arguments

- sentiment_data:

  Data frame with 'sentiment' column

- category_var:

  Name of the category variable column

- plot_type:

  Type of plot: "bar" or "stacked" (default: "bar")

- title:

  Plot title (default: auto-generated)

## Value

A plotly grouped/stacked bar chart

## Examples

``` r
if (FALSE) { # \dontrun{
data <- data.frame(
  text = c("Good", "Bad", "Okay", "Great", "Poor"),
  category = c("A", "A", "B", "B", "B")
)
data <- cbind(data, analyze_sentiment(data$text))
plot <- plot_sentiment_by_category(data, "category")
print(plot)
} # }
```
