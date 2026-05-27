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

A ggplot2 grouped/stacked bar chart

## Examples

``` r
# \donttest{
articles <- TextAnalysisR::SpecialEduTech[1:20, ]
sentiment_results <- analyze_sentiment(articles$abstract)
sentiment_data <- cbind(
  reference_type = articles$reference_type,
  sentiment_results
)
sentiment_plot <- plot_sentiment_by_category(sentiment_data, "reference_type")
print(sentiment_plot)

# }
```
