# Calculate Dispersion Metrics

Computes quantitative dispersion metrics for terms, measuring how evenly
distributed they are across the corpus.

## Usage

``` r
calculate_dispersion_metrics(tokens_object, terms)
```

## Arguments

- tokens_object:

  A quanteda tokens object

- terms:

  Character vector of terms to analyze

## Value

Data frame with columns:

- term: The search term

- frequency: Total occurrences

- doc_count: Number of documents containing term

- doc_ratio: Proportion of documents containing term

- juilland_d: Juilland's D dispersion (0-1, higher = more even)

- rosengren_s: Rosengren's S dispersion
