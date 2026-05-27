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

## Examples

``` r
# \donttest{
abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:20]
term_matrix <- as.matrix(quanteda::dfm(quanteda::tokens(abstracts)))
kmeans_result <- stats::kmeans(term_matrix, centers = 2)
metrics <- calculate_clustering_metrics(kmeans_result$cluster, term_matrix)
print(metrics)
#> $n_clusters
#> [1] 2
#> 
#> $cluster_sizes
#> clusters
#>  1  2 
#> 14  6 
#> 
#> $silhouette
#> [1] 0.4675644
#> 
#> $davies_bouldin
#> [1] 1.024449
#> 
#> $calinski_harabasz
#> [1] 16.50688
#> 
# }
```
