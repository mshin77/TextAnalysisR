# Auto-tune BERTopic Hyperparameters

Automatically searches for optimal hyperparameters for embedding-based
topic modeling. Evaluates multiple configurations of UMAP and HDBSCAN
parameters and returns the best model based on the specified metric.
Embeddings are generated once and reused across all configurations for
efficiency.

## Usage

``` r
auto_tune_embedding_topics(
  texts,
  embeddings = NULL,
  embedding_model = "all-MiniLM-L6-v2",
  n_trials = 12,
  metric = "silhouette",
  seed = 123,
  verbose = TRUE
)
```

## Arguments

- texts:

  Character vector of documents to analyze.

- embeddings:

  Precomputed embeddings matrix (optional). If NULL, embeddings are
  generated.

- embedding_model:

  Embedding model name (default: "all-MiniLM-L6-v2").

- n_trials:

  Maximum number of configurations to try (default: 12).

- metric:

  Optimization metric: "silhouette", "coherence", or "combined"
  (default: "silhouette").

- seed:

  Random seed for reproducibility.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing:

- best_config: Data frame with the optimal hyperparameter configuration

- best_model: The topic model fitted with optimal parameters

- all_results: List of all evaluated configurations with metrics

- n_trials_completed: Number of configurations successfully evaluated

## Details

The function searches over these parameters:

- n_neighbors: UMAP neighborhood size (5, 10, 15, 25)

- min_cluster_size: HDBSCAN minimum cluster size (3, 5, 10)

- cluster_selection_method: "eom" (broader) or "leaf" (finer-grained)

## See also

Other topic-modeling:
[`analyze_semantic_evolution()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_semantic_evolution.md),
[`assess_embedding_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_embedding_stability.md),
[`assess_hybrid_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_hybrid_stability.md),
[`calculate_assignment_consistency()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_assignment_consistency.md),
[`calculate_eval_metrics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_eval_metrics_internal.md),
[`calculate_keyword_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_keyword_stability.md),
[`calculate_semantic_drift()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_semantic_drift.md),
[`calculate_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_probability.md),
[`calculate_topic_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_stability.md),
[`find_optimal_k()`](https://mshin77.github.io/TextAnalysisR/reference/find_optimal_k.md),
[`find_topic_matches()`](https://mshin77.github.io/TextAnalysisR/reference/find_topic_matches.md),
[`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md),
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
if (FALSE) { # \dontrun{
  texts <- c("Machine learning for image recognition",
             "Deep learning neural networks",
             "Natural language processing models",
             "Computer vision applications")

  tuning_result <- auto_tune_embedding_topics(
    texts = texts,
    n_trials = 6,
    metric = "silhouette",
    verbose = TRUE
  )

  # View best configuration
  tuning_result$best_config

  # Use the best model
  best_model <- tuning_result$best_model
} # }
```
