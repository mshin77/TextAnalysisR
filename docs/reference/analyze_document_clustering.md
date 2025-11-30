# Analyze Document Clustering

Complete document clustering analysis with dimensionality reduction and
optional clustering

## Usage

``` r
analyze_document_clustering(
  feature_matrix,
  method = "UMAP",
  clustering_method = "none",
  ...
)
```

## Arguments

- feature_matrix:

  Feature matrix (documents x features)

- method:

  Dimensionality reduction method ("PCA", "t-SNE", "UMAP")

- clustering_method:

  Clustering method ("none", "kmeans", "hierarchical", "dbscan",
  "hdbscan")

- ...:

  Additional parameters for methods

## Value

List containing coordinates, clusters, method info, and quality metrics
