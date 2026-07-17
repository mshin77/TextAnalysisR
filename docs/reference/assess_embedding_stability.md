# Assess Embedding Topic Model Stability

Evaluates the stability of embedding-based topic modeling by running
multiple models with different random seeds and comparing their results.
High stability (high ARI, consistent keywords) indicates stable topic
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

  Number of model runs with different seeds (default: 5). More runs give
  a steadier stability estimate.

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

## Examples

``` r
if (interactive()) {
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
}
```
