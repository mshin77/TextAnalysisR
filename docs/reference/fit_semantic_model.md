# Fit Semantic Model

Performs comprehensive semantic analysis including similarity,
dimensionality reduction, and clustering. This is a high-level wrapper
function.

## Usage

``` r
fit_semantic_model(
  texts,
  analysis_types = c("similarity", "dimensionality_reduction", "clustering"),
  document_feature_type = "embeddings",
  similarity_method = "cosine",
  use_embeddings = TRUE,
  embedding_model = "all-MiniLM-L6-v2",
  dimred_method = "UMAP",
  clustering_method = "umap_dbscan",
  n_components = 2,
  n_clusters = 5,
  seed = 123,
  verbose = TRUE
)
```

## Arguments

- texts:

  A character vector of texts to analyze.

- analysis_types:

  Types of analysis to perform: "similarity",
  "dimensionality_reduction", "clustering".

- document_feature_type:

  Feature extraction type (default: "embeddings").

- similarity_method:

  Similarity calculation method (default: "cosine").

- use_embeddings:

  Logical, use embedding-based approaches (default: TRUE).

- embedding_model:

  Sentence transformer model name (default: "all-MiniLM-L6-v2").

- dimred_method:

  Dimensionality reduction method: "PCA", "t-SNE", "UMAP" (default:
  "UMAP").

- clustering_method:

  Clustering method: "kmeans", "hierarchical", "umap_dbscan" (default:
  "umap_dbscan").

- n_components:

  Number of dimensions for reduction (default: 2).

- n_clusters:

  Number of clusters (default: 5).

- seed:

  Random seed for reproducibility (default: 123).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing results from requested analyses.

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
  data(SpecialEduTech)
  texts <- SpecialEduTech$abstract[1:10]

  results <- fit_semantic_model(
    texts = texts,
    analysis_types = c("similarity", "clustering")
  )

  print(results)
}
```
