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

## See also

Other sentiment:
[`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md),
[`plot_document_sentiment_trajectory()`](https://mshin77.github.io/TextAnalysisR/reference/plot_document_sentiment_trajectory.md),
[`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md),
[`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md),
[`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md),
[`plot_sentiment_violin()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_violin.md),
[`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md),
[`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c("Great results!", "Poor performance", "Okay outcome")
sentiment_data <- analyze_sentiment(texts)
plot <- plot_sentiment_distribution(sentiment_data)
print(plot)
} # }
```
