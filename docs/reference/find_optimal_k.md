# Find Optimal Number of Topics

Searches for the optimal number of topics (K) using stm::searchK.
Produces diagnostic plots to help select the best K value.

## Usage

``` r
find_optimal_k(
  dfm_object,
  topic_range,
  max.em.its = 75,
  emtol = 1e-04,
  cores = 1,
  categorical_var = NULL,
  continuous_var = NULL,
  height = 600,
  width = 800,
  verbose = TRUE,
  ...
)
```

## Arguments

- dfm_object:

  A quanteda dfm object to be used for topic modeling.

- topic_range:

  A vector of K values to test (e.g., 2:10).

- max.em.its:

  Maximum number of EM iterations (default: 75).

- emtol:

  Convergence tolerance for EM algorithm (default: 1e-04). Higher values
  (e.g., 1e-03) speed up fitting but may reduce precision.

- cores:

  Number of CPU cores to use for parallel processing (default: 1). Set
  to higher values for faster searchK on multi-core systems.

- categorical_var:

  Optional categorical variable(s) for prevalence.

- continuous_var:

  Optional continuous variable(s) for prevalence.

- height:

  Plot height in pixels (default: 600).

- width:

  Plot width in pixels (default: 800).

- verbose:

  Logical indicating whether to print progress (default: TRUE).

- ...:

  Additional arguments passed to stm::searchK.

## Value

A list containing search results and diagnostic plots.

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
