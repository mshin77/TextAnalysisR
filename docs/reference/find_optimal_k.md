# Find Optimal Number of Topics

Searches for the optimal number of topics (K) using stm::searchK.
Produces diagnostic plots to help select the best K value. The default
spectral initialization is deterministic; with LDA or random
initialization, fit several seeds per K before selecting a model.

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

[`plot_quality_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/plot_quality_metrics.md)
to visualize topic-count diagnostics;
[`stm::stm()`](https://rdrr.io/pkg/stm/man/stm.html) to fit the chosen
model;
[`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md)
for an embedding-based alternative to STM
