# Contrastive Learning Topic Modeling

Implements contrastive learning approaches for topic modeling to improve
topic separation and discriminability.

## Usage

``` r
run_contrastive_topics_internal(
  texts,
  n_topics = 10,
  temperature = 0.1,
  negative_sampling_rate = 5,
  embedding_model = "all-MiniLM-L6-v2",
  seed = 123
)
```

## Arguments

- texts:

  Character vector of documents

- n_topics:

  Number of topics to discover

- temperature:

  Temperature parameter for contrastive learning

- negative_sampling_rate:

  Rate of negative sampling

- embedding_model:

  Transformer model for embeddings

- seed:

  Random seed for reproducibility

## Value

List containing contrastive topic model and metrics

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
[`run_neural_topics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/run_neural_topics_internal.md),
[`run_temporal_topics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/run_temporal_topics_internal.md),
[`validate_semantic_coherence()`](https://mshin77.github.io/TextAnalysisR/reference/validate_semantic_coherence.md)
