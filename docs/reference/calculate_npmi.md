# Calculate NPMI for Topic Top-Terms

NPMI (Normalized Pointwise Mutual Information) measures coherence of
top-K terms per topic using internal document co-occurrence from the
supplied DFM (boolean presence).

## Usage

``` r
calculate_npmi(top_terms_list, dfm, top_k = 10, epsilon = 1e-12)
```

## Arguments

- top_terms_list:

  Named or unnamed list of character vectors, one per topic.

- dfm:

  A quanteda dfm providing the reference doc-term frequencies.

- top_k:

  Number of top terms per topic to include (default 10).

- epsilon:

  Numerical stabilizer (default 1e-12).

## Value

list with `mean_npmi` (scalar) and `per_topic_npmi` (numeric vector).
