# Assess Hybrid Model Stability via Bootstrap

Evaluates the stability of a hybrid topic model by running bootstrap
resampling. This helps identify which topics are robust and which may be
artifacts of the specific sample. Based on research recommendations for
topic model validation.

## Usage

``` r
assess_hybrid_stability(
  texts,
  n_topics = 10,
  n_bootstrap = 5,
  sample_proportion = 0.8,
  embedding_model = "all-MiniLM-L6-v2",
  seed = 123,
  verbose = TRUE
)
```

## Arguments

- texts:

  A character vector of texts to analyze.

- n_topics:

  Number of topics (default: 10).

- n_bootstrap:

  Number of bootstrap iterations (default: 5).

- sample_proportion:

  Proportion of documents to sample (default: 0.8).

- embedding_model:

  Embedding model name (default: "all-MiniLM-L6-v2").

- seed:

  Random seed for reproducibility.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing stability metrics:

- topic_stability: Per-topic stability scores (0-1)

- mean_stability: Overall stability score

- keyword_stability: Stability of top keywords per topic

- alignment_stability: Stability of STM-embedding alignment

- bootstrap_results: Detailed results from each bootstrap run

## See also

Other topic-modeling:
[`analyze_semantic_evolution()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_semantic_evolution.md),
[`calculate_assignment_consistency()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_assignment_consistency.md),
[`calculate_eval_metrics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_eval_metrics_internal.md),
[`calculate_keyword_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_keyword_stability.md),
[`calculate_semantic_drift()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_semantic_drift.md),
[`calculate_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_probability.md),
[`calculate_topic_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_stability.md),
[`find_optimal_k()`](https://mshin77.github.io/TextAnalysisR/reference/find_optimal_k.md),
[`find_topic_matches()`](https://mshin77.github.io/TextAnalysisR/reference/find_topic_matches.md),
[`fit_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_topics.md),
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
  stability <- assess_hybrid_stability(
    texts = my_texts,
    n_topics = 10,
    n_bootstrap = 5,
    verbose = TRUE
  )

  # View topic stability scores
  stability$topic_stability
} # }
```
