# Analyze Sentiment Using Tidytext Lexicons

Performs lexicon-based sentiment analysis on a DFM object using tidytext
lexicons. Supports AFINN, Bing, and NRC lexicons with scoring and
emotion analysis. Now supports n-grams for improved negation and
intensifier handling.

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

  A quanteda DFM object (unigram or n-gram)

- lexicon:

  Lexicon to use: "afinn", "bing", or "nrc" (default: "bing")

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
# \donttest{
abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
corpus <- quanteda::corpus(abstracts)
dfm_object <- quanteda::dfm(quanteda::tokens(corpus))
lexicon_results <- sentiment_lexicon_analysis(dfm_object, lexicon = "bing")
print(lexicon_results$document_sentiment)
#> # A tibble: 10 × 6
#>    document positive negative sentiment_score total_sentiment_words sentiment
#>    <chr>       <dbl>    <dbl>           <dbl>                 <dbl> <chr>    
#>  1 text1           3        0               3                     3 positive 
#>  2 text10          0        3              -3                     3 negative 
#>  3 text2           1        1               0                     2 neutral  
#>  4 text3           1        1               0                     2 neutral  
#>  5 text4           2        1               1                     3 positive 
#>  6 text5           2        0               2                     2 positive 
#>  7 text6           7        6               1                    13 positive 
#>  8 text7           1        1               0                     2 neutral  
#>  9 text8          12        1              11                    13 positive 
#> 10 text9           3        4              -1                     7 negative 
# }
```
