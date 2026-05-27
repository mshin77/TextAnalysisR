# Plot Sentiment Box Plot by Category

Creates a box plot showing sentiment score distribution by category.

## Usage

``` r
plot_sentiment_boxplot(
  sentiment_data,
  category_var = "category_var",
  title = "Sentiment Score Distribution"
)
```

## Arguments

- sentiment_data:

  Data frame from analyze_sentiment() containing sentiment_score and
  category columns

- category_var:

  Name of the category variable column (default: "category_var")

- title:

  Plot title (default: "Sentiment Score Distribution")

## Value

A ggplot2 box plot
