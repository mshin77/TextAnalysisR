# Dimensionality Reduction Analysis

This function performs dimensionality reduction using various methods
including PCA, t-SNE, and UMAP. For efficiency and consistency, PCA
preprocessing is always performed first, and t-SNE/UMAP use the PCA
results as input. This follows best practices for high-dimensional data
analysis.

## Usage

``` r
reduce_dimensions(
  data_matrix,
  method = "PCA",
  n_components = 2,
  pca_dims = 50,
  tsne_perplexity = 30,
  tsne_max_iter = 1000,
  umap_neighbors = 15,
  umap_min_dist = 0.1,
  umap_metric = "cosine",
  seed = 123,
  verbose = TRUE
)
```

## Arguments

- data_matrix:

  A numeric matrix where rows represent documents and columns represent
  features.

- method:

  The dimensionality reduction method. Options: "PCA", "t-SNE", "UMAP".

- n_components:

  The number of components/dimensions to reduce to (default: 2).

- pca_dims:

  The number of dimensions for PCA preprocessing (default: 50).

- tsne_perplexity:

  The perplexity parameter for t-SNE (default: 30).

- tsne_max_iter:

  The maximum number of iterations for t-SNE (default: 1000).

- umap_neighbors:

  The number of neighbors for UMAP (default: 15).

- umap_min_dist:

  The minimum distance for UMAP (default: 0.1).

- umap_metric:

  The metric for UMAP (default: "cosine").

- seed:

  Random seed for reproducibility (default: 123).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing the reduced dimensions, method used, and additional
metadata.

## See also

Other semantic:
[`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md),
[`analyze_similarity_gaps()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md),
[`calculate_clustering_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_clustering_metrics.md),
[`calculate_cross_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_cross_similarity.md),
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
[`semantic_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_document_clustering.md),
[`semantic_similarity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_similarity_analysis.md),
[`temporal_semantic_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/temporal_semantic_analysis.md),
[`validate_cross_models()`](https://mshin77.github.io/TextAnalysisR/reference/validate_cross_models.md),
[`word_co_occurrence_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_co_occurrence_network.md),
[`word_correlation_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_correlation_network.md)

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech

  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )

  tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")

  dfm_object <- quanteda::dfm(tokens)

  data_matrix <- as.matrix(dfm_object)

  pca_result <- TextAnalysisR::reduce_dimensions(
    data_matrix,
    method = "PCA"
  )
  print(pca_result)

  tsne_result <- TextAnalysisR::reduce_dimensions(
    data_matrix,
    method = "t-SNE"
  )
  print(tsne_result)

  umap_result <- TextAnalysisR::reduce_dimensions(
    data_matrix,
    method = "UMAP"
  )
  print(umap_result)
}
```
