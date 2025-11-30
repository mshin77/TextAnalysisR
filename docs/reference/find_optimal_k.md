# Find Optimal Number of Topics

Searches for the optimal number of topics (K) using stm::searchK.
Produces diagnostic plots to help select the best K value.

## Usage

``` r
find_optimal_k(
  dfm_object,
  topic_range,
  max.em.its = 75,
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
