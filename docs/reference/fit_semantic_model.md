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

## Examples

``` r
if (interactive()) {
  texts <- c(
    "Assistive technology supports learning.",
    "Technology aids students with disabilities.",
    "Machine learning improves predictions."
  )

  results <- fit_semantic_model(
    texts = texts,
    analysis_types = c("similarity", "clustering")
  )

  print(results)
}
```
