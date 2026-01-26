# Embedding-based Topic Modeling (Deprecated)

This function is deprecated. Please use
[`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md)
instead.

## Usage

``` r
fit_embedding_topics(
  texts,
  method = "umap_hdbscan",
  n_topics = 10,
  embedding_model = "all-MiniLM-L6-v2",
  clustering_method = "kmeans",
  similarity_threshold = 0.7,
  min_topic_size = 3,
  cluster_selection_method = "eom",
  umap_neighbors = 15,
  umap_min_dist = 0,
  umap_n_components = 5,
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
