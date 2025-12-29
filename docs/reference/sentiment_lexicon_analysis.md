# Analyze Sentiment Using Tidytext Lexicons

Performs lexicon-based sentiment analysis on a DFM object using tidytext
lexicons. Supports AFINN, Bing, and NRC lexicons with comprehensive
scoring and emotion analysis. Now supports n-grams for improved negation
and intensifier handling.

## Usage

``` r
sentiment_lexicon_analysis(
  dfm_object,
  lexicon = "afinn",
  texts_df = NULL,
  feature_type = "words",
  ngram_range = 2,
  texts = NULL
)
```

## Arguments

- dfm_object:

  A quanteda DFM object (unigram or n-gram)

- lexicon:

  Lexicon to use: "afinn", "bing", or "nrc" (default: "afinn")

- texts_df:

  Optional data frame with original texts and metadata (default: NULL)

- feature_type:

  Feature space: "words" (unigrams) or "ngrams" (default: "words")

- ngram_range:

  N-gram size when feature_type = "ngrams" (default: 2 for bigrams)

- texts:

  Optional character vector of texts for n-gram creation (default: NULL)

## Value

A list containing:

- document_sentiment:

  Data frame with sentiment scores per document

- emotion_scores:

  Data frame with emotion scores (NRC only)

- summary_stats:

  List of summary statistics

- feature_type:

  Feature type used for analysis

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
[`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md)

## Examples

``` r
if (FALSE) { # \dontrun{
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:10]
corp <- quanteda::corpus(texts)
dfm_obj <- quanteda::dfm(quanteda::tokens(corp))
results <- sentiment_lexicon_analysis(dfm_obj, lexicon = "afinn")
print(results$document_sentiment)
} # }
```
