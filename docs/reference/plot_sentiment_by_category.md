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

## See also

Other sentiment:
[`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md),
[`analyze_sentiment_llm()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment_llm.md),
[`plot_document_sentiment_trajectory()`](https://mshin77.github.io/TextAnalysisR/reference/plot_document_sentiment_trajectory.md),
[`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md),
[`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md),
[`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md),
[`plot_sentiment_violin()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_violin.md),
[`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md),
[`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)

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
