# Analyze Text Sentiment

Performs sentiment analysis on text data using the syuzhet package.
Returns sentiment scores and classifications.

## Usage

``` r
analyze_sentiment(texts, method = "syuzhet", doc_ids = NULL)
```

## Arguments

- texts:

  Character vector of texts to analyze

- method:

  Sentiment analysis method: "syuzhet", "bing", "afinn", or "nrc"
  (default: "syuzhet")

- doc_ids:

  Optional character vector of document identifiers (default: NULL)

## Value

A data frame with columns:

- document:

  Document identifier

- text:

  Original text

- sentiment_score:

  Numeric sentiment score

- sentiment:

  Classification: "positive", "negative", or "neutral"

## See also

Other sentiment:
[`analyze_sentiment_llm()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment_llm.md),
[`plot_document_sentiment_trajectory()`](https://mshin77.github.io/TextAnalysisR/reference/plot_document_sentiment_trajectory.md),
[`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md),
[`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md),
[`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md),
[`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md),
[`plot_sentiment_violin()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_violin.md),
[`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md),
[`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)

## Examples

``` r
if (FALSE) { # \dontrun{
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:10]
results <- analyze_sentiment(texts)
print(results)
} # }
```
