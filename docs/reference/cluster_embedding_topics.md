# Embedding-Based Topic Discovery

Groups documents into topics by clustering transformer embeddings, with
per-topic cohesion diagnostics.

## Usage

``` r
cluster_embedding_topics(
  texts,
  n_topics = 10,
  embedding_model = "all-MiniLM-L6-v2",
  clustering_method = c("kmeans", "hierarchical"),
  min_topic_size = 3,
  seed = 123
)
```

## Arguments

- texts:

  Character vector of documents

- n_topics:

  Number of topics to discover

- embedding_model:

  Transformer model for initial embeddings

- clustering_method:

  Algorithm applied to the embedding similarity matrix: "kmeans"
  (default) or "hierarchical". Both honour `n_topics` and assign every
  document.

- min_topic_size:

  Minimum documents per topic.

- seed:

  Random seed for reproducibility

## Value

List with topic assignments and diagnostics. The cohesion values are
mean pairwise cosine similarity of document embeddings within each topic
cluster (embedding-space compactness), not lexical coherence measures
such as C_v or NPMI.

## See also

[`find_optimal_k()`](https://mshin77.github.io/TextAnalysisR/reference/find_optimal_k.md)
and
[`auto_tune_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/auto_tune_embedding_topics.md)
for choosing `n_topics`;
[`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md)
for the UMAP and HDBSCAN pipeline that derives the topic count from
density instead.
