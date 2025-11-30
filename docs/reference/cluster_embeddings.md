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

- seed:

  Random seed for reproducibility (default: 123).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing cluster assignments, method used, and quality metrics.

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
