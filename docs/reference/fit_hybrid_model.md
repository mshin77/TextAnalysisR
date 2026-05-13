# Fit Hybrid Topic Model

Fits a hybrid topic model combining STM with embedding-based methods.
This function integrates structural topic modeling (STM) with semantic
embeddings for enhanced topic discovery. The STM component provides
statistical rigor and covariate modeling capabilities, while the
embedding component adds semantic coherence.

**Effect Estimation:** Covariate effects on topic prevalence can be
estimated using the STM component via
[`stm::estimateEffect()`](https://rdrr.io/pkg/stm/man/estimateEffect.html).
The embedding component provides semantically meaningful topic
representations but does not support direct covariate modeling.

## Usage

``` r
fit_hybrid_model(
  texts,
  metadata = NULL,
  n_topics_stm = 10,
  embedding_model = "all-MiniLM-L6-v2",
  stm_prevalence = NULL,
  stm_init_type = "Spectral",
  stm_gamma_prior = "Pooled",
  stm_kappa_prior = "L1",
  stm_max_em_its = 75,
  compute_quality = TRUE,
  stm_weight = 0.5,
  n_embedding_dims = 5,
  precomputed_embeddings = NULL,
  verbose = TRUE,
  seed = 123
)
```

## Arguments

- texts:

  A character vector of texts to analyze.

- metadata:

  Optional data frame with document metadata for STM covariate modeling.

- n_topics_stm:

  Number of topics for STM (default: 10).

- embedding_model:

  Embedding model name (default: "all-MiniLM-L6-v2").

- stm_prevalence:

  Formula for STM prevalence covariates (e.g., ~ category + s(year,
  df=3)).

- stm_init_type:

  STM initialization type (default: "Spectral").

- stm_gamma_prior:

  Prior for STM topic-covariate coefficients, passed to
  `stm::stm(gamma.prior = ...)` (default: "Pooled").

- stm_kappa_prior:

  Prior for STM topic-word distributions, passed to
  `stm::stm(kappa.prior = ...)` (default: "L1").

- stm_max_em_its:

  Maximum EM iterations for STM (default: 75).

- compute_quality:

  Logical, if TRUE, computes quality metrics (default: TRUE).

- stm_weight:

  Weight for STM in keyword combination, 0-1 (default: 0.5).

- n_embedding_dims:

  UMAP output dimensions for the embedding pipeline, forwarded to
  `fit_embedding_model(umap_n_components = ...)` (default: 5).

- precomputed_embeddings:

  Optional numeric matrix of precomputed document embeddings (one row
  per text). If provided, skips embedding generation.

- verbose:

  Logical, if TRUE, prints progress messages.

- seed:

  Random seed for reproducibility.

## Value

A list containing:

- stm_result: The STM model output (use this for effect estimation)

- embedding_result: The embedding-based topic model output

- alignment: Alignment metrics including cosine similarity, assignment
  agreement, correlation, and Adjusted Rand Index

- quality_metrics: Quality metrics including coherence, exclusivity,
  silhouette scores, and combined quality score

- combined_topics: Integrated topic representations with weighted
  keywords

- stm_data: STM-formatted data (needed for effect estimation)

- metadata: Metadata used in modeling

## Note

For covariate effect estimation, use
[`stm::estimateEffect()`](https://rdrr.io/pkg/stm/man/estimateEffect.html)
on the `stm_result$model` component with `stm_data$meta` as the
metadata.

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
[`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md),
[`run_neural_topics_internal()`](https://mshin77.github.io/TextAnalysisR/reference/run_neural_topics_internal.md),
[`validate_semantic_coherence()`](https://mshin77.github.io/TextAnalysisR/reference/validate_semantic_coherence.md)

## Examples

``` r
if (interactive()) {
  texts <- c("Computer-assisted instruction improves math skills for students with disabilities",
             "Assistive technology supports reading comprehension for learning disabled students",
             "Mobile devices enhance communication for students with autism spectrum disorder")

  hybrid_model <- fit_hybrid_model(
    texts = texts,
    n_topics_stm = 3,
    compute_quality = TRUE,
    verbose = TRUE
  )

  # View alignment metrics
  hybrid_model$alignment$overall_alignment
  hybrid_model$alignment$adjusted_rand_index

  # View quality metrics
  hybrid_model$quality_metrics$stm_coherence_mean
  hybrid_model$quality_metrics$combined_quality

  # View combined keywords with source attribution
  hybrid_model$combined_topics[[1]]$combined_keywords
}
```
