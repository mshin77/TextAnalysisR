# Estimate topic prevalence effects from an STM model

Estimates how a document-level covariate shifts topic prevalence,
returning a tidy data frame of per-topic estimates with intervals.

## Usage

``` r
estimate_topic_effects(
  estimates,
  variable,
  type = c("pointestimate", "continuous"),
  method = c("stm", "beta"),
  interval = NULL,
  model = NULL,
  documents = NULL,
  npoints = 100,
  nsims = 100,
  ci = 0.95
)
```

## Arguments

- estimates:

  An `estimateEffect` object from
  [`stm::estimateEffect`](https://rdrr.io/pkg/stm/man/estimateEffect.html).

- variable:

  Name of the covariate to evaluate.

- type:

  "pointestimate" for a categorical covariate or "continuous" for a
  numeric one.

- method:

  "stm" (method of composition, default) or "beta" (bounded per-topic
  Beta regression).

- interval:

  "eti" equal-tailed (default for "stm") or "hpd"
  highest-posterior-density (default for "beta").

- model:

  Fitted `stm` model; required for `method = "beta"`.

- documents:

  Document list passed to `stm`; required for `method = "beta"`.

- npoints:

  Grid resolution for a continuous covariate (default 100).

- nsims:

  Posterior draws (default 100; 25 for `method = "beta"`).

- ci:

  Interval width (default 0.95).

## Value

A data frame with columns topic, value, proportion, lower, and upper.
