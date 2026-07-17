# Calculate Topic Stability

Calculates stability of topics across consecutive time periods. Each
period is fitted independently, so topics are matched one-to-one on
keyword Jaccard similarity before scoring (see
[`calculate_keyword_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_keyword_stability.md)).

## Usage

``` r
calculate_topic_stability(temporal_results)
```

## Arguments

- temporal_results:

  Results from temporal analysis.

## Value

Stability metrics: matched-pair stability per consecutive-period
transition and their mean.
