# Fit Topic Prevalence Model

Fits a count regression model to topic prevalence data, auto-selecting
between Poisson, Negative Binomial, and Zero-Inflated Negative Binomial
based on dispersion ratio and zero-inflation diagnostics.

## Usage

``` r
fit_topic_prevalence_model(
  topic_proportions,
  metadata,
  formula,
  model_type = "auto",
  zero_inflation_threshold = 0.5,
  count_multiplier = 1000,
  max_iterations = 200
)
```

## Arguments

- topic_proportions:

  Numeric vector of topic proportions (0-1) for one topic.

- metadata:

  Data frame of document-level covariates.

- formula:

  Model formula (character or formula object). Response variable is
  created internally as `topic_count`.

- model_type:

  Model selection strategy: `"auto"` (default), `"poisson"`, `"negbin"`,
  or `"zeroinfl"`.

- zero_inflation_threshold:

  Proportion of zeros above which a zero-inflated model is attempted
  (default: 0.5).

- count_multiplier:

  Multiplier to convert proportions to pseudo-counts (default: 1000).

- max_iterations:

  Maximum iterations for model fitting (default: 200).

## Value

List containing:

- `model`: Fitted model object

- `summary`: Tidy summary with odds ratios

- `model_type`: Selected model type

- `diagnostics`: Zero proportion, dispersion ratio, mean/variance

- `formula`: Formula used

## See also

Other topic-modeling:
[`analyze_semantic_evolution()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_semantic_evolution.md),
[`assess_embedding_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_embedding_stability.md),
[`auto_tune_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/auto_tune_embedding_topics.md),
[`calculate_assignment_consistency()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_assignment_consistency.md),
[`calculate_keyword_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_keyword_stability.md),
[`calculate_npmi()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_npmi.md),
[`calculate_semantic_drift()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_semantic_drift.md),
[`calculate_topic_diversity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_diversity.md),
[`calculate_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_probability.md),
[`calculate_topic_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_stability.md),
[`extract_topic_terms_df()`](https://mshin77.github.io/TextAnalysisR/reference/extract_topic_terms_df.md),
[`find_optimal_k()`](https://mshin77.github.io/TextAnalysisR/reference/find_optimal_k.md),
[`find_topic_matches()`](https://mshin77.github.io/TextAnalysisR/reference/find_topic_matches.md),
[`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md),
[`fit_temporal_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_temporal_model.md),
[`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md),
[`get_topic_prevalence()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_prevalence.md),
[`get_topic_terms()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_terms.md),
[`get_topic_texts()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_texts.md),
[`identify_topic_trends()`](https://mshin77.github.io/TextAnalysisR/reference/identify_topic_trends.md),
[`plot_model_comparison()`](https://mshin77.github.io/TextAnalysisR/reference/plot_model_comparison.md),
[`plot_quality_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/plot_quality_metrics.md),
[`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md),
[`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md),
[`plot_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_probability.md),
[`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md),
[`run_neural_topics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/run_neural_topics_internal.md),
[`validate_semantic_coherence()`](https://mshin77.github.io/TextAnalysisR/reference/validate_semantic_coherence.md)
