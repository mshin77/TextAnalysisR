# Embedding-based Document Clustering

This function performs clustering analysis using various methods,
ordered from simple to comprehensive: k-means (simplest), hierarchical
(intermediate), and UMAP+DBSCAN (most comprehensive).

## Usage

``` r
cluster_embeddings(
  data_matrix,
  method = "kmeans",
  n_clusters = 0,
  umap_neighbors = 15,
  umap_min_dist = 0.1,
  umap_n_components = 10,
  umap_metric = "cosine",
  dbscan_eps = 0,
  dbscan_min_samples = 5,
  reduce_outliers = TRUE,
  outlier_strategy = "centroid",
  seed = 123,
  verbose = TRUE
)
```

## Arguments

- data_matrix:

  A numeric matrix where rows represent documents and columns represent
  features.

- method:

  The clustering method. Options: "kmeans", "hierarchical",
  "umap_dbscan".

- n_clusters:

  The number of clusters (for k-means and hierarchical). If 0, optimal
  number is determined automatically.

- umap_neighbors:

  The number of neighbors for UMAP (default: 15).

- umap_min_dist:

  The minimum distance for UMAP (default: 0.1).

- umap_n_components:

  The number of UMAP components (default: 10).

- umap_metric:

  The metric for UMAP (default: "cosine").

- dbscan_eps:

  The eps parameter for DBSCAN. If 0, optimal value is determined
  automatically.

- dbscan_min_samples:

  The minimum samples for DBSCAN (default: 5).

- reduce_outliers:

  Logical, if TRUE, reassigns noise points (cluster 0) to nearest
  cluster (default: TRUE).

- outlier_strategy:

  Strategy for outlier reduction: "centroid" (default, Euclidean
  distance in UMAP space) or "embeddings" (cosine similarity in original
  space). Follows BERTopic methodology.

- seed:

  Random seed for reproducibility (default: 123).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing cluster assignments, method used, and quality metrics.

## See also

Other semantic:
[`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md),
[`analyze_similarity_gaps()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md),
[`calculate_clustering_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_clustering_metrics.md),
[`calculate_cross_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_cross_similarity.md),
[`calculate_document_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_document_similarity.md),
[`calculate_similarity_robust()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_similarity_robust.md),
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

  kmeans_result <- TextAnalysisR::cluster_embeddings(
    data_matrix,
    method = "kmeans",
    n_clusters = 5
  )
  print(kmeans_result)

  hierarchical_result <- TextAnalysisR::cluster_embeddings(
    data_matrix,
    method = "hierarchical",
    n_clusters = 5
  )
  print(hierarchical_result)

  umap_dbscan_result <- TextAnalysisR::cluster_embeddings(
    data_matrix,
    method = "umap_dbscan",
    umap_neighbors = 15,
    umap_min_dist = 0.1
  )
  print(umap_dbscan_result)
}
```
