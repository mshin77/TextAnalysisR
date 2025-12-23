# Calculate Clustering Quality Metrics

Calculates common clustering evaluation metrics including Silhouette
Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

## Usage

``` r
calculate_clustering_metrics(
  clusters,
  data_matrix,
  dist_matrix = NULL,
  metrics = "all"
)
```

## Arguments

- clusters:

  Integer vector of cluster assignments

- data_matrix:

  Numeric matrix of data points (rows = observations, cols = features)

- dist_matrix:

  Optional distance matrix. If NULL, computed from data_matrix

- metrics:

  Character vector of metrics to calculate. Options: "silhouette",
  "davies_bouldin", "calinski_harabasz", or "all" (default)

## Value

A named list containing:

- silhouette:

  Silhouette score (-1 to 1, higher is better)

- davies_bouldin:

  Davies-Bouldin index (lower is better)

- calinski_harabasz:

  Calinski-Harabasz index (higher is better)

- n_clusters:

  Number of clusters

- cluster_sizes:

  Table of cluster sizes

## Details

- Silhouette Score: Measures how similar an object is to its own cluster
  compared to other clusters. Range: -1 to 1, higher is better.

- Davies-Bouldin Index: Average similarity between each cluster and its
  most similar cluster. Lower values indicate better clustering.

- Calinski-Harabasz Index: Ratio of between-cluster to within-cluster
  variance. Higher values indicate better-defined clusters.

## See also

Other semantic:
[`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md),
[`analyze_similarity_gaps()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md),
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
[`reduce_dimensions()`](https://mshin77.github.io/TextAnalysisR/reference/reduce_dimensions.md),
[`semantic_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_document_clustering.md),
[`semantic_similarity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_similarity_analysis.md),
[`temporal_semantic_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/temporal_semantic_analysis.md),
[`validate_cross_models()`](https://mshin77.github.io/TextAnalysisR/reference/validate_cross_models.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Generate sample data
set.seed(123)
data <- rbind(
  matrix(rnorm(100, mean = 0), ncol = 2),
  matrix(rnorm(100, mean = 3), ncol = 2)
)
clusters <- c(rep(1, 50), rep(2, 50))

metrics <- calculate_clustering_metrics(clusters, data)
print(metrics)
} # }
```
