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
