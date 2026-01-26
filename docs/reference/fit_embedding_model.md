# Fit Embedding-based Topic Model

This function performs embedding-based topic modeling using transformer
embeddings and specialized clustering techniques. The primary method
uses the BERTopic library, which combines transformer embeddings with
UMAP dimensionality reduction and HDBSCAN clustering for optimal topic
discovery. This approach creates more semantically coherent topics
compared to traditional methods by leveraging deep learning embeddings.

## Usage

``` r
fit_embedding_model(
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

  The topic modeling method: "umap_hdbscan" (uses BERTopic),
  "embedding_clustering", "hierarchical_semantic".

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

## See also

Other topic-modeling:
[`analyze_semantic_evolution()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_semantic_evolution.md),
[`assess_embedding_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_embedding_stability.md),
[`assess_hybrid_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_hybrid_stability.md),
[`auto_tune_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/auto_tune_embedding_topics.md),
[`calculate_assignment_consistency()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_assignment_consistency.md),
[`calculate_eval_metrics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_eval_metrics_internal.md),
[`calculate_keyword_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_keyword_stability.md),
[`calculate_semantic_drift()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_semantic_drift.md),
[`calculate_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_probability.md),
[`calculate_topic_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_stability.md),
[`find_optimal_k()`](https://mshin77.github.io/TextAnalysisR/reference/find_optimal_k.md),
[`find_topic_matches()`](https://mshin77.github.io/TextAnalysisR/reference/find_topic_matches.md),
[`fit_hybrid_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_hybrid_model.md),
[`fit_temporal_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_temporal_model.md),
[`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md),
[`get_topic_prevalence()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_prevalence.md),
[`get_topic_terms()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_terms.md),
[`get_topic_texts()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_texts.md),
[`identify_topic_trends()`](https://mshin77.github.io/TextAnalysisR/reference/identify_topic_trends.md),
[`plot_model_comparison()`](https://mshin77.github.io/TextAnalysisR/reference/plot_model_comparison.md),
[`plot_quality_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/plot_quality_metrics.md),
[`run_contrastive_topics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/run_contrastive_topics_internal.md),
[`run_neural_topics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/run_neural_topics_internal.md),
[`run_temporal_topics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/run_temporal_topics_internal.md),
[`validate_semantic_coherence()`](https://mshin77.github.io/TextAnalysisR/reference/validate_semantic_coherence.md)

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
