# Calculate Keyword Stability

Calculates stability between the topic keyword sets of two independently
fitted models. Topic numbering is arbitrary across fits, so topics are
matched one-to-one by greedy assignment on the pairwise keyword Jaccard
matrix before averaging matched-pair similarity.

## Usage

``` r
calculate_keyword_stability(keywords1, keywords2)
```

## Arguments

- keywords1:

  List of keyword vectors, one per topic, from the first model.

- keywords2:

  List of keyword vectors, one per topic, from the second model.

## Value

Mean Jaccard similarity of matched topic pairs (0-1).
