# Plot Emotion Radar Chart

Creates a polar/radar chart for NRC emotion analysis with optional
grouping.

## Usage

``` r
plot_emotion_radar(
  emotion_data,
  group_var = NULL,
  normalize = FALSE,
  title = "Emotion Analysis"
)
```

## Arguments

- emotion_data:

  Data frame with emotion scores (columns: emotion, total_score)

- group_var:

  Optional grouping variable column name for overlaid radars (default:
  NULL)

- normalize:

  Logical, whether to normalize scores to 0-100 scale (default: FALSE)

- title:

  Plot title (default: "Emotion Analysis")

## Value

A plotly polar chart

## See also

Other sentiment:
[`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md),
[`analyze_sentiment_llm()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment_llm.md),
[`plot_document_sentiment_trajectory()`](https://mshin77.github.io/TextAnalysisR/reference/plot_document_sentiment_trajectory.md),
[`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md),
[`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md),
[`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md),
[`plot_sentiment_violin()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_violin.md),
[`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md),
[`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)
