# Generate Cluster Labels

Generate descriptive labels for document clusters

## Usage

``` r
generate_cluster_labels_auto(
  feature_matrix,
  clusters,
  method = "tfidf",
  n_terms = 3
)
```

## Arguments

- feature_matrix:

  Feature matrix used for clustering

- clusters:

  Cluster assignments

- method:

  Label generation method ("tfidf", "representative", "frequent")

- n_terms:

  Number of terms per label

## Value

Named list of cluster labels
