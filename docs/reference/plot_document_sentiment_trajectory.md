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

## See also

Other sentiment:
[`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md),
[`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md),
[`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md),
[`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md),
[`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md),
[`plot_sentiment_violin()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_violin.md),
[`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md),
[`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)
