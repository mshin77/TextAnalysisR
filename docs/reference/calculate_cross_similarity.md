# Calculate Cross-Matrix Cosine Similarity

Calculates cosine similarity between two different embedding matrices,
useful for comparing documents/topics across different categories or
groups.

## Usage

``` r
calculate_cross_similarity(
  embeddings1,
  embeddings2,
  labels1 = NULL,
  labels2 = NULL,
  normalize = TRUE
)
```

## Arguments

- embeddings1:

  A numeric matrix where rows are items and columns are embedding
  dimensions.

- embeddings2:

  A numeric matrix where rows are items and columns are embedding
  dimensions. Must have the same number of columns as embeddings1.

- labels1:

  Optional character vector of labels for items in embeddings1.

- labels2:

  Optional character vector of labels for items in embeddings2.

- normalize:

  Logical, whether to L2-normalize embeddings before computing
  similarity (default: TRUE).

## Value

A list containing:

- similarity_matrix:

  Matrix of cosine similarities (nrow(embeddings1) x nrow(embeddings2))

- similarity_df:

  Long-format data frame with columns: row_idx, col_idx, similarity, and
  optionally label1, label2

## Examples

``` r
# \donttest{
abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:6]
term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(abstracts)))
similarity_result <- calculate_cross_similarity(
  term_matrix[1:3, ], term_matrix[4:6, ],
  labels1 = paste("Doc", 1:3),
  labels2 = paste("Doc", 4:6)
)
similarity_result$similarity_matrix
#>        docs
#> docs        text4     text5     text6
#>   text1 0.3947055 0.4779596 0.6083902
#>   text2 0.1842614 0.3396613 0.3482887
#>   text3 0.1362998 0.2680007 0.2795068
# }
```
