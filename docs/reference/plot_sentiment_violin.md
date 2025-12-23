# Plot Sentiment Violin Plot by Category

Creates a violin plot showing sentiment score distribution by category.

## Usage

``` r
plot_sentiment_violin(
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

A plotly violin plot

## See also

Other sentiment:
[`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md),
[`plot_document_sentiment_trajectory()`](https://mshin77.github.io/TextAnalysisR/reference/plot_document_sentiment_trajectory.md),
[`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md),
[`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md),
[`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md),
[`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md),
[`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md),
[`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)
