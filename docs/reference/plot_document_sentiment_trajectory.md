# Plot Document Sentiment Trajectory

Creates a line chart showing sentiment scores across documents with
color gradient.

## Usage

``` r
plot_document_sentiment_trajectory(
  sentiment_data,
  top_n = NULL,
  doc_ids = NULL,
  text_preview = NULL,
  title = "Document Sentiment Scores"
)
```

## Arguments

- sentiment_data:

  Data frame from analyze_sentiment() with sentiment_score column

- top_n:

  Number of documents to display (default: NULL for all)

- doc_ids:

  Optional vector of custom document IDs for display (default: NULL)

- text_preview:

  Optional vector of text snippets for tooltips (default: NULL)

- title:

  Plot title (default: "Document Sentiment Scores")

## Value

A plotly line chart with color gradient
