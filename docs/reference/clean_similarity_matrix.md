# Clean Similarity Matrix

Cleans a similarity matrix by handling NA/Inf values, ensuring symmetry,
and setting diagonal to 1.

## Usage

``` r
clean_similarity_matrix(similarity_matrix)
```

## Arguments

- similarity_matrix:

  A numeric matrix of similarity values.

## Value

Cleaned similarity matrix.

## See also

Other matrix-utilities:
[`renumber_clusters_sequentially()`](https://mshin77.github.io/TextAnalysisR/reference/renumber_clusters_sequentially.md)
