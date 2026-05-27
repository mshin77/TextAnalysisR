# Calculate Topic Diversity (Unique-Word Proportion)

Diversity = unique top-K terms across all topics / (n_topics \* top_k).
Range 0 (all topics identical) to 1 (fully disjoint). Complementary to
NPMI coherence: a good model scores high on both.

## Usage

``` r
calculate_topic_diversity(top_terms_list, top_k = 25)
```

## Arguments

- top_terms_list:

  Named or unnamed list of character vectors, one per topic.

- top_k:

  Number of top terms per topic to include (default 25).

## Value

Numeric between 0 and 1, or NA if input empty.
