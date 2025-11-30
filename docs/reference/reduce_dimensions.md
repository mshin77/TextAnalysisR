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
