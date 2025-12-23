# Generate Topic Labels Using OpenAI's API

This function generates descriptive labels for each topic based on their
top terms using OpenAI's ChatCompletion API.

## Usage

``` r
generate_topic_labels(
  top_topic_terms,
  model = "gpt-3.5-turbo",
  system = NULL,
  user = NULL,
  temperature = 0.5,
  openai_api_key = NULL,
  verbose = TRUE
)
```

## Arguments

- top_topic_terms:

  A data frame containing the top terms for each topic.

- model:

  A character string specifying which OpenAI model to use (default:
  "gpt-3.5-turbo").

- system:

  A character string containing the system prompt for the OpenAI API. If
  NULL, the function uses the default system prompt.

- user:

  A character string containing the user prompt for the OpenAI API. If
  NULL, the function uses the default user prompt.

- temperature:

  A numeric value controlling the randomness of the output (default:
  0.5).

- openai_api_key:

  A character string containing the OpenAI API key. If NULL, the
  function attempts to load the key from the OPENAI_API_KEY environment
  variable or the .env file in the working directory.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A data frame containing the top terms for each topic along with their
generated labels.

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

  tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")

  dfm_object <- quanteda::dfm(tokens)

  out <- quanteda::convert(dfm_object, to = "stm")

stm_15 <- stm::stm(
  data = out$meta,
  documents = out$documents,
  vocab = out$vocab,
  max.em.its = 75,
  init.type = "Spectral",
  K = 15,
  prevalence = ~ reference_type + s(year),
  verbose = TRUE)

top_topic_terms <- TextAnalysisR::get_topic_terms(
  stm_model = stm_15,
  top_term_n = 10,
  verbose = TRUE
  )

top_labeled_topic_terms <- TextAnalysisR::generate_topic_labels(
  top_topic_terms,
  model = "gpt-3.5-turbo",
  temperature = 0.5,
  openai_api_key = "your_openai_api_key",
  verbose = TRUE)
print(top_labeled_topic_terms)

top_labeled_topic_terms <- TextAnalysisR::generate_topic_labels(
  top_topic_terms,
  model = "gpt-3.5-turbo",
  temperature = 0.5,
  verbose = TRUE)
print(top_labeled_topic_terms)
}
```
