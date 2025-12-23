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

## See also

Other semantic:
[`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md),
[`analyze_similarity_gaps()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md),
[`calculate_clustering_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_clustering_metrics.md),
[`calculate_document_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_document_similarity.md),
[`calculate_similarity_robust()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_similarity_robust.md),
[`cluster_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/cluster_embeddings.md),
[`cross_analysis_validation()`](https://mshin77.github.io/TextAnalysisR/reference/cross_analysis_validation.md),
[`export_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/export_document_clustering.md),
[`extract_cross_category_similarities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_cross_category_similarities.md),
[`fit_semantic_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_semantic_model.md),
[`generate_cluster_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels.md),
[`generate_cluster_labels_auto()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels_auto.md),
[`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md),
[`reduce_dimensions()`](https://mshin77.github.io/TextAnalysisR/reference/reduce_dimensions.md),
[`semantic_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_document_clustering.md),
[`semantic_similarity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_similarity_analysis.md),
[`temporal_semantic_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/temporal_semantic_analysis.md),
[`validate_cross_models()`](https://mshin77.github.io/TextAnalysisR/reference/validate_cross_models.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Generate embeddings for two groups
emb1 <- TextAnalysisR::generate_embeddings(c("text a", "text b"), verbose = FALSE)
emb2 <- TextAnalysisR::generate_embeddings(c("text c", "text d", "text e"), verbose = FALSE)

# Calculate cross-similarity
result <- calculate_cross_similarity(
  emb1, emb2,
  labels1 = c("A", "B"),
  labels2 = c("C", "D", "E")
)
print(result$similarity_matrix)
print(result$similarity_df)
} # }
```
