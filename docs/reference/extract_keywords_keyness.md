# Extract Keywords Using Statistical Keyness

Extracts distinctive keywords by comparing document groups using
log-likelihood ratio (G-squared).

## Usage

``` r
extract_keywords_keyness(dfm, target, top_n = 20, measure = "lr")
```

## Arguments

- dfm:

  A quanteda dfm object

- target:

  Target document indices or logical vector

- top_n:

  Number of top keywords to extract (default: 20)

- measure:

  Keyness measure: "lr" (log-likelihood) or "chi2" (default: "lr")

## Value

Data frame with columns: Keyword, Keyness_Score

## Examples

``` r
if (FALSE) { # \dontrun{
library(quanteda)
corp <- corpus(c("positive text", "negative text", "positive words"))
dfm_obj <- dfm(tokens(corp))
# Compare first document vs rest
keywords <- extract_keywords_keyness(dfm_obj, target = 1)
print(keywords)
} # }
```
