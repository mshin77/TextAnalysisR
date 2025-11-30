# Plot Sentiment Distribution

Creates a bar plot showing the distribution of sentiment
classifications.

## Usage

``` r
plot_sentiment_distribution(sentiment_data, title = "Sentiment Distribution")
```

## Arguments

- sentiment_data:

  Data frame from analyze_sentiment() or with 'sentiment' column

- title:

  Plot title (default: "Sentiment Distribution")

## Value

A plotly bar chart

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c("Great results!", "Poor performance", "Okay outcome")
sentiment_data <- analyze_sentiment(texts)
plot <- plot_sentiment_distribution(sentiment_data)
print(plot)
} # }
```
