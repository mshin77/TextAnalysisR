# Embedding-based Sentiment Analysis

Performs sentiment analysis using transformer-based embeddings and
neural models. This approach uses pre-trained language models for
contextual sentiment detection without requiring sentiment lexicons.
Particularly effective for handling:

- Complex contextual sentiment

- Implicit sentiment and sarcasm

- Domain-specific sentiment

- Negation and intensifiers (automatically handled by the model)

## Usage

``` r
sentiment_embedding_analysis(
  texts,
  embeddings = NULL,
  model_name = "distilbert-base-uncased-finetuned-sst-2-english",
  doc_names = NULL,
  use_gpu = FALSE
)
```

## Arguments

- texts:

  Character vector of texts to analyze

- embeddings:

  Optional pre-computed embedding matrix (from generate_embeddings)

- model_name:

  Sentiment model name (default:
  "distilbert-base-uncased-finetuned-sst-2-english")

- doc_names:

  Optional document names/IDs

- use_gpu:

  Whether to use GPU if available (default: FALSE)

## Value

A list containing:

- document_sentiment:

  Data frame with document-level sentiment scores and classifications

- emotion_scores:

  NULL (emotion detection not currently supported for embeddings)

- summary_stats:

  Summary statistics including document counts and average scores

- model_used:

  Name of the transformer model used

- feature_type:

  "embeddings"

## See also

Other sentiment:
[`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md),
[`analyze_sentiment_llm()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment_llm.md),
[`plot_document_sentiment_trajectory()`](https://mshin77.github.io/TextAnalysisR/reference/plot_document_sentiment_trajectory.md),
[`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md),
[`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md),
[`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md),
[`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md),
[`plot_sentiment_violin()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_violin.md),
[`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)

## Examples

``` r
if (FALSE) { # \dontrun{
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:10]
result <- sentiment_embedding_analysis(texts)
print(result$document_sentiment)
print(result$summary_stats)
} # }
```
