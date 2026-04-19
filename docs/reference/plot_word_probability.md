# Plot Word Probabilities by Topic

Creates a faceted bar plot showing the top terms and their probabilities
(beta values) for each topic in a topic model.

## Usage

``` r
plot_word_probability(
  top_topic_terms,
  topic_label = NULL,
  ncol = 3,
  height = 1200,
  width = 800,
  ylab = "Word probability",
  title = NULL,
  colors = NULL,
  measure_label = "Beta",
  base_font_size = 11,
  ...
)
```

## Arguments

- top_topic_terms:

  A data frame containing topic terms with columns: topic, term, and
  beta.

- topic_label:

  Optional topic labels. Can be either a named vector mapping topic
  numbers to labels, or a character string specifying a column name in
  top_topic_terms (default: NULL).

- ncol:

  Number of columns for facet wrap layout (default: 3).

- height:

  Plot height for responsive spacing adjustments (default: 1200).

- width:

  Plot width for responsive spacing adjustments (default: 800).

- ylab:

  Y-axis label (default: "Word probability").

- title:

  Plot title (default: NULL for auto-generated title).

- colors:

  Color palette for topics (default: NULL for auto-generated colors).

- measure_label:

  Label for the probability measure (default: "Beta").

- base_font_size:

  Base font size in points for the plot theme (default: 11). Axis text
  and strip text will be base_font_size + 2.

- ...:

  Additional arguments (currently unused, kept for compatibility).

## Value

A ggplot2 object showing word probabilities faceted by topic.

## See also

Other topic-modeling:
[`analyze_semantic_evolution()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_semantic_evolution.md),
[`assess_embedding_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_embedding_stability.md),
[`assess_hybrid_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_hybrid_stability.md),
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
[`fit_hybrid_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_hybrid_model.md),
[`fit_temporal_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_temporal_model.md),
[`fit_topic_prevalence_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_topic_prevalence_model.md),
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
[`run_neural_topics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/run_neural_topics_internal.md),
[`validate_semantic_coherence()`](https://mshin77.github.io/TextAnalysisR/reference/validate_semantic_coherence.md)
