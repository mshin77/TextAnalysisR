# Calculate Log Odds Ratio Between Categories

Computes log odds ratio to compare word frequencies between categories.
Identifies words that are distinctively used in one category vs another.
Uses Laplace smoothing to handle zero counts.

## Usage

``` r
calculate_log_odds_ratio(
  dfm_object,
  group_var,
  comparison_mode = c("binary", "one_vs_rest", "pairwise"),
  reference_level = NULL,
  top_n = 10,
  min_count = 5
)
```

## Arguments

- dfm_object:

  A quanteda dfm object

- group_var:

  Character, name of the grouping variable in docvars

- comparison_mode:

  Character, one of "binary", "one_vs_rest", or "pairwise"

  - binary: Compare two categories directly

  - one_vs_rest: Compare each category against all others combined

  - pairwise: Compare all pairs of categories

- reference_level:

  Character, reference category for binary comparison (default: first
  level)

- top_n:

  Number of top terms per comparison (default: 10)

- min_count:

  Minimum word count to include (default: 5)

## Value

Data frame with columns:

- term: The word/feature

- category1: First category in comparison

- category2: Second category in comparison

- count1: Count in category 1

- count2: Count in category 2

- odds1: Odds in category 1

- odds2: Odds in category 2

- odds_ratio: Ratio of odds

- log_odds_ratio: Log of odds ratio (positive = more in compared
  category)

## Examples

``` r
# \donttest{
articles <- TextAnalysisR::SpecialEduTech[1:20, ]
corpus <- quanteda::corpus(
  articles$abstract,
  docvars = data.frame(reference_type = articles$reference_type)
)
dfm_object <- quanteda::dfm(quanteda::tokens(corpus))
log_odds <- calculate_log_odds_ratio(dfm_object, "reference_type")
# }
```
