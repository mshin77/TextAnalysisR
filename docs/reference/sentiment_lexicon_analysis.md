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

## Examples

``` r
if (FALSE) { # \dontrun{
corp <- quanteda::corpus(c("I love this!", "I hate that", "It's okay"))
dfm_obj <- quanteda::dfm(quanteda::tokens(corp))
results <- sentiment_lexicon_analysis(dfm_obj, lexicon = "afinn")
print(results$document_sentiment)

texts <- c("not good at all", "very happy indeed")
results_ngram <- sentiment_lexicon_analysis(
  dfm_obj,
  lexicon = "bing",
  feature_type = "ngrams",
  ngram_range = 2,
  texts = texts
)
} # }
```
