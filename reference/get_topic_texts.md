# Convert Topic Terms to Text Strings

Concatenates top terms for each topic into text strings suitable for
embedding generation. Useful for creating topic representations for
semantic similarity analysis.

## Usage

``` r
get_topic_texts(
  top_terms_df,
  topic_var = "topic",
  term_var = "term",
  weight_var = NULL,
  sep = " ",
  top_n = NULL
)
```

## Arguments

- top_terms_df:

  A data frame containing top terms for topics, typically output from
  [`get_topic_terms`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_terms.md).

- topic_var:

  Name of the column containing topic identifiers (default: "topic").

- term_var:

  Name of the column containing terms (default: "term").

- weight_var:

  Optional name of column with term weights (e.g., "beta"). If provided,
  terms are ordered by weight before concatenation.

- sep:

  Separator between terms (default: " ").

- top_n:

  Optional number of top terms to include per topic (default: NULL, uses
  all).

## Value

A character vector of topic text strings, one per topic, ordered by
topic number.

## See also

Other topic-modeling:
[`analyze_semantic_evolution()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_semantic_evolution.md),
[`assess_hybrid_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_hybrid_stability.md),
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
# Get topic terms from STM model
top_terms <- TextAnalysisR::get_topic_terms(stm_model, top_term_n = 10)

# Convert to text strings for embedding
topic_texts <- get_topic_texts(top_terms)

# Generate embeddings
topic_embeddings <- TextAnalysisR::generate_embeddings(topic_texts)
} # }
```
