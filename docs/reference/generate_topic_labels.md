# Generate Topic Labels Using AI

This function generates descriptive labels for each topic based on their
top terms using AI providers (OpenAI, Gemini, or Ollama).

## Usage

``` r
generate_topic_labels(
  top_topic_terms,
  provider = "auto",
  model = NULL,
  system = NULL,
  user = NULL,
  temperature = 0.5,
  api_key = NULL,
  openai_api_key = NULL,
  verbose = TRUE
)
```

## Arguments

- top_topic_terms:

  A data frame containing the top terms for each topic.

- provider:

  AI provider to use: "auto" (default), "openai", "gemini", or "ollama".
  "auto" will try Ollama first, then check for OpenAI/Gemini keys.

- model:

  A character string specifying which model to use. If NULL, uses
  provider defaults: "gpt-4.1-mini" (OpenAI), "gemini-2.5-flash"
  (Gemini), or recommended Ollama model.

- system:

  A character string containing the system prompt for the API. If NULL,
  the function uses the default system prompt.

- user:

  A character string containing the user prompt for the API. If NULL,
  the function uses the default user prompt.

- temperature:

  A numeric value controlling the randomness of the output (default:
  0.5).

- api_key:

  API key for OpenAI or Gemini. If NULL, uses environment variable. Not
  required for Ollama.

- openai_api_key:

  Deprecated. Use `api_key` instead. Kept for backward compatibility.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A data frame containing the top terms for each topic along with their
generated labels.

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

## Examples

``` r
if (interactive()) {
top_topic_terms <- get_topic_terms(stm_model, top_term_n = 10)

# Auto-detect provider (tries Ollama -> OpenAI -> Gemini)
labels <- generate_topic_labels(top_topic_terms)

# Use specific provider
labels_ollama <- generate_topic_labels(top_topic_terms, provider = "ollama")
labels_openai <- generate_topic_labels(top_topic_terms, provider = "openai")
labels_gemini <- generate_topic_labels(top_topic_terms, provider = "gemini")
}
```
