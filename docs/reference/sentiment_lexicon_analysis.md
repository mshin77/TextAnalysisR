# Analyze Sentiment Using Tidytext Lexicons

Performs lexicon-based sentiment analysis on a DFM object using tidytext
lexicons. Supports AFINN, Bing, and NRC lexicons with scoring and
emotion analysis.

Tokens absent from the lexicon are ignored (not treated as neutral);
`summary_stats$token_match_rate` reports the share of corpus tokens the
lexicon covered. AFINN classification uses a +/- 0.5 band on the
per-sentiment-word average (`avg_sentiment`); its `n_sentiment_words`
column counts matched sentiment tokens, not document length.

## Usage

``` r
sentiment_lexicon_analysis(
  dfm_object,
  lexicon = "bing",
  texts_df = NULL,
  feature_type = "words",
  ngram_range = 2,
  texts = NULL
)
```

## Arguments

- dfm_object:

  A quanteda DFM object (unigram)

- lexicon:

  Lexicon to use: "afinn", "bing", or "nrc" (default: "bing")

- texts_df:

  Optional data frame with original texts and metadata (default: NULL)

- feature_type:

  Feature space: "words" (unigrams). The AFINN/Bing/NRC lexicons are
  unigram lexicons; "ngrams" falls back to unigram scoring with a
  warning (default: "words").

- ngram_range:

  Retained for backward compatibility; not used for scoring (default: 2)

- texts:

  Optional character vector of texts used to rebuild a unigram DFM
  (default: NULL)

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
# \donttest{
abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
corpus <- quanteda::corpus(abstracts)
dfm_object <- quanteda::dfm(quanteda::tokens(corpus))
lexicon_results <- sentiment_lexicon_analysis(dfm_object, lexicon = "bing")
print(lexicon_results$document_sentiment)
#> # A tibble: 10 × 7
#>    document positive negative total_sentiment_words sentiment_score
#>    <chr>       <dbl>    <dbl>                 <dbl>           <dbl>
#>  1 text1           3        0                     3          1     
#>  2 text10          0        3                     3         -1     
#>  3 text2           1        1                     2          0     
#>  4 text3           1        1                     2          0     
#>  5 text4           2        1                     3          0.333 
#>  6 text5           2        0                     2          1     
#>  7 text6           7        6                    13          0.0769
#>  8 text7           1        1                     2          0     
#>  9 text8          12        1                    13          0.846 
#> 10 text9           3        4                     7         -0.143 
#> # ℹ 2 more variables: sentiment_raw <dbl>, sentiment <chr>
# }
```
