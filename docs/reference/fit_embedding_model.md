# Fit Embedding-based Topic Model

This function performs embedding-based topic modeling using transformer
embeddings and specialized clustering techniques. Supports two backends:

- **Python backend** (default): Uses BERTopic library which combines
  transformer embeddings with UMAP dimensionality reduction and HDBSCAN
  clustering for optimal topic discovery.

- **R backend**: Uses R-native packages (umap, dbscan, Rtsne) for users
  without Python/BERTopic installed. Provides similar functionality with
  c-TF-IDF keyword extraction.

## Usage

``` r
fit_embedding_model(
  texts,
  method = "umap_hdbscan",
  n_topics = 10,
  embedding_model = "all-MiniLM-L6-v2",
  backend = "auto",
  clustering_method = "kmeans",
  similarity_threshold = 0.7,
  min_topic_size = 10,
  cluster_selection_method = "eom",
  umap_neighbors = 15,
  umap_min_dist = 0,
  umap_n_components = 5,
  umap_metric = "cosine",
  tsne_perplexity = 30,
  pca_dims = 50,
  dbscan_eps = 0.5,
  dbscan_minpts = 5,
  representation_method = "c-tfidf",
  diversity = 0.5,
  reduce_outliers = TRUE,
  outlier_strategy = "probabilities",
  outlier_threshold = 0,
  seed = 123,
  verbose = TRUE,
  precomputed_embeddings = NULL
)
```

## Arguments

- texts:

  A character vector of texts to analyze.

- method:

  The topic modeling method:

  - For Python backend: "umap_hdbscan" (uses BERTopic)

  - For R backend: "umap_dbscan", "umap_kmeans", "umap_hierarchical",
    "tsne_dbscan", "tsne_kmeans", "pca_kmeans", "pca_hierarchical"

  - For both: "embedding_clustering", "hierarchical_semantic"

- n_topics:

  The number of topics to identify. For UMAP+HDBSCAN, use NULL or "auto"
  for automatic determination, or specify an integer.

- embedding_model:

  The embedding model to use (default: "all-MiniLM-L6-v2").

- backend:

  The backend to use: "auto" (default, tries Python then R), "python"
  (requires BERTopic), or "r" (R-native packages only).

- clustering_method:

  The clustering method for embedding-based approach: "kmeans",
  "hierarchical", "dbscan", "hdbscan".

- similarity_threshold:

  The similarity threshold for topic assignment (default: 0.7).

- min_topic_size:

  The minimum number of documents per topic (default: 3).

- cluster_selection_method:

  HDBSCAN cluster selection method: "eom" (Excess of Mass, default) or
  "leaf" (finer-grained topics).

- umap_neighbors:

  The number of neighbors for UMAP dimensionality reduction (default:
  15).

- umap_min_dist:

  The minimum distance for UMAP (default: 0.0). Use 0.0 for tight,
  well-separated clusters. Use 0.1+ for visualization purposes. Range:
  0.0-0.99.

- umap_n_components:

  The number of UMAP components (default: 5).

- umap_metric:

  Distance metric for UMAP: "cosine" (recommended for text) or
  "euclidean" (default: "cosine").

- tsne_perplexity:

  Perplexity parameter for t-SNE (default: 30). Only used when method
  includes "tsne".

- pca_dims:

  Number of PCA components for dimensionality reduction (default: 50).
  Only used when method includes "pca".

- dbscan_eps:

  Epsilon parameter for DBSCAN (default: 0.5). Neighborhood size for
  density-based clustering.

- dbscan_minpts:

  Minimum points for DBSCAN core points (default: 5).

- representation_method:

  The method for topic representation: "c-tfidf", "tfidf", "mmr",
  "frequency" (default: "c-tfidf").

- diversity:

  Topic diversity parameter between 0 and 1 (default: 0.5).

- reduce_outliers:

  Logical, if TRUE, reduces outliers in HDBSCAN clustering (default:
  TRUE).

- outlier_strategy:

  Strategy for outlier reduction using BERTopic: "probabilities"
  (default, uses topic probabilities), "c-tf-idf" (uses c-TF-IDF
  similarity), "embeddings" (uses cosine similarity in embedding space),
  or "distributions" (uses topic distributions). Ignored if
  reduce_outliers = FALSE.

- outlier_threshold:

  Minimum threshold for outlier reassignment (default: 0.0). Higher
  values require stronger evidence for reassignment.

- seed:

  Random seed for reproducibility (default: 123).

- verbose:

  Logical, if TRUE, prints progress messages.

- precomputed_embeddings:

  Optional matrix of pre-computed document embeddings. If provided,
  skips embedding generation for improved performance. Must have the
  same number of rows as the length of texts.

## Value

A list containing topic assignments, topic keywords, and quality
metrics.

## See also

[`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md)
to supply precomputed embeddings;
[`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md)
for AI-suggested topic names;
[`find_optimal_k()`](https://mshin77.github.io/TextAnalysisR/reference/find_optimal_k.md)
for an STM-based alternative

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech
  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )
  texts <- united_tbl$united_texts

  # Embedding-based topic modeling (powered by BERTopic)
  result <- TextAnalysisR::fit_embedding_model(
    texts = texts,
    method = "umap_hdbscan",
    n_topics = 8,
    min_topic_size = 3
  )

  print(result$topic_assignments)
  print(result$topic_keywords)
}
```
