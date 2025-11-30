# Plot Top Documents by Sentiment Score

Creates a bar plot of documents with highest/lowest sentiment scores.

## Usage

``` r
plot_top_sentiment_documents(
  sentiment_data,
  top_n = 15,
  order = "highest",
  title = NULL
)
```

## Arguments

- sentiment_data:

  Data frame from analyze_sentiment()

- top_n:

  Number of documents to show (default: 15)

- order:

  Direction: "highest" or "lowest" (default: "highest")

- title:

  Plot title (default: auto-generated)

## Value

A plotly bar chart
