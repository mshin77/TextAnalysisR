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
# \donttest{
abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
dfm_object <- quanteda::dfm(quanteda::tokens(quanteda::corpus(abstracts)))
keywords <- extract_keywords_keyness(dfm_object, target = 1)
print(keywords)
#>              Keyword Keyness_Score
#> 1           approach      8.225973
#> 2                alp      5.789556
#> 3  learning-disabled      5.789556
#> 4               when      2.364533
#> 5         advantages      1.337108
#> 6            advised      1.337108
#> 7           although      1.337108
#> 8            caution      1.337108
#> 9           choosing      1.337108
#> 10     circumstances      1.337108
#> 11          clinical      1.337108
#> 12          concepts      1.337108
#> 13          deficits      1.337108
#> 14       demonstrate      1.337108
#> 15         fractions      1.337108
#> 16               has      1.337108
#> 17              have      1.337108
#> 18       instigation      1.337108
#> 19               its      1.337108
#> 20            memory      1.337108
# }
```
