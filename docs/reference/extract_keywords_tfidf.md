# Extract Keywords Using TF-IDF

Extracts top keywords from a document-feature matrix using TF-IDF
weighting.

## Usage

``` r
extract_keywords_tfidf(dfm, top_n = 20, normalize = FALSE)
```

## Arguments

- dfm:

  A quanteda dfm object

- top_n:

  Number of top keywords to extract (default: 20)

- normalize:

  Logical, whether to normalize TF-IDF scores to 0-1 range (default:
  FALSE)

## Value

Data frame with columns: Keyword, TF_IDF_Score, Frequency

## Examples

``` r
# \donttest{
  library(quanteda)
#> Package version: 4.4
#> Unicode version: 15.1
#> ICU version: 74.1
#> Parallel computing: 8 of 8 threads used.
#> See https://quanteda.io for tutorials and examples.
  corp <- corpus(c("text analysis", "data mining", "text mining"))
  dfm_obj <- dfm(tokens(corp))
  keywords <- extract_keywords_tfidf(dfm_obj, top_n = 5)
  print(keywords)
#>    Keyword TF_IDF_Score Frequency
#> 1 analysis    0.4771213         1
#> 2     data    0.4771213         1
#> 3     text    0.3521825         2
#> 4   mining    0.3521825         2
# }
```
