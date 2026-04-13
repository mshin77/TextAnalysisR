# Calculate Weighted Log Odds Ratio

Computes weighted log odds ratios using the method from Monroe,
Colaresi, and Quinn (2008) "Fightin' Words" via the tidylo package. This
method weights log odds by variance (z-score) to identify words that
reliably distinguish between groups, accounting for sampling
variability.

## Usage

``` r
calculate_weighted_log_odds(dfm_object, group_var, top_n = 10, min_count = 5)
```

## Arguments

- dfm_object:

  A quanteda dfm object

- group_var:

  Character, name of the document variable to group by

- top_n:

  Number of top terms to return per group (default: 10)

- min_count:

  Minimum total count for a term to be included (default: 5)

## Value

A data frame with columns: group, feature, n, log_odds_weighted, and
log_odds (from tidylo::bind_log_odds)

## References

Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words:
Lexical feature selection and evaluation for identifying the content of
political conflict. Political Analysis, 16(4), 372-403.

Silge, J., & Robinson, D. (2017). Text mining with R: A tidy approach.
O'Reilly Media. https://www.tidytextmining.com/

## Examples

``` r
if (FALSE) { # \dontrun{
weighted_odds <- calculate_weighted_log_odds(dfm, "party", top_n = 15)
} # }
```
