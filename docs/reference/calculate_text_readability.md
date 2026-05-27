# Calculate Text Readability

Calculates multiple readability metrics for texts including Flesch
Reading Ease, Flesch-Kincaid Grade Level, Gunning FOG index, and others.
Optionally includes lexical diversity metrics and sentence statistics.

## Usage

``` r
calculate_text_readability(
  texts,
  metrics = c("flesch", "flesch_kincaid", "gunning_fog"),
  include_lexical_diversity = TRUE,
  include_sentence_stats = TRUE,
  dfm_for_lexdiv = NULL,
  doc_names = NULL
)
```

## Arguments

- texts:

  Character vector of texts to analyze

- metrics:

  Character vector of readability metrics to calculate. Options:
  "flesch", "flesch_kincaid", "gunning_fog", "smog", "ari",
  "coleman_liau"

- include_lexical_diversity:

  Logical, include TTR and MTLD (default: TRUE)

- include_sentence_stats:

  Logical, include average sentence length (default: TRUE)

- dfm_for_lexdiv:

  Optional pre-computed DFM for lexical diversity calculation

- doc_names:

  Optional character vector of document names

## Value

A data frame with document names and readability scores

## Examples

``` r
# \donttest{
data(SpecialEduTech, package = "TextAnalysisR")
texts <- SpecialEduTech$abstract[1:10]
readability <- calculate_text_readability(texts)
print(readability)
#>    Document     flesch flesch_kincaid gunning_fog Lexical Diversity (TTR)
#> 1     Doc 1  19.824902       17.53294    21.21569               0.7831325
#> 2     Doc 2  -5.024231       20.41923    24.24615               0.9166667
#> 3     Doc 3   5.505682       16.59045    22.35758               0.7812500
#> 4     Doc 4  27.617151       14.74849    17.90233               0.9069767
#> 5     Doc 5  16.490000       17.17000    20.00000               0.5757576
#> 6     Doc 6  14.513863       16.77798    21.36197               0.4415205
#> 7     Doc 7 -10.401667       19.18185    24.23704               0.8823529
#> 8     Doc 8  26.795461       15.57450    19.74545               0.4392157
#> 9     Doc 9  15.338100       16.87522    18.47530               0.5882353
#> 10   Doc 10  -2.587500       19.58250    22.93333               0.8510638
#>    Avg Sentence Length
#> 1             27.33333
#> 2             24.00000
#> 3             16.00000
#> 4             21.50000
#> 5             24.75000
#> 6             21.68750
#> 7             17.00000
#> 8             18.85714
#> 9             22.50000
#> 10            23.50000
# }
```
