# Assess Embedding Topic Model Stability

Evaluates the stability of embedding-based topic modeling by running
multiple models with different random seeds and comparing their results.
High stability (high ARI, consistent keywords) indicates robust topic
structure in the data.

## Usage

``` r
assess_embedding_stability(
  texts,
  n_runs = 5,
  embedding_model = "all-MiniLM-L6-v2",
  select_best = TRUE,
  base_seed = 123,
  verbose = TRUE,
  ...
)
```

## Arguments

- texts:

  Character vector of documents to analyze.

- n_runs:

  Number of model runs with different seeds (default: 5).

- embedding_model:

  Embedding model name (default: "all-MiniLM-L6-v2").

- select_best:

  Logical, if TRUE, returns the best model by quality (default: TRUE).

- base_seed:

  Base random seed; each run uses base_seed + (run - 1).

- verbose:

  Logical, if TRUE, prints progress messages.

- ...:

  Additional arguments passed to fit_embedding_model().

## Value

A list containing:

- stability_metrics: List with mean_ari, sd_ari, mean_jaccard,
  quality_variance

- best_model: Best model by silhouette score (if select_best = TRUE)

- all_models: List of all fitted models

- is_stable: Logical, TRUE if mean ARI \>= 0.6 (considered stable)

- recommendation: Text recommendation based on stability

## Details

Stability is assessed via:

- Adjusted Rand Index (ARI): Measures agreement in topic assignments
  across runs

- Keyword Jaccard similarity: Measures overlap in top keywords per topic

- Quality variance: Variance in silhouette scores across runs

## See also

Other topic-modeling:
[`analyze_semantic_evolution()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_semantic_evolution.md),
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
             "Natural language processing models")

  stability <- assess_embedding_stability(
    texts = texts,
    n_runs = 3,
    verbose = TRUE
  )

  # Check if results are stable
  stability$is_stable
  stability$stability_metrics$mean_ari

  # Use the best model
  best_model <- stability$best_model
} # }
```
