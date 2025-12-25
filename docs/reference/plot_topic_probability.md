# Plot Per-Document Per-Topic Probabilities

Generates a bar plot showing the prevalence of each topic across all
documents.

## Usage

``` r
plot_topic_probability(
  stm_model = NULL,
  gamma_data = NULL,
  top_n = 10,
  height = 800,
  width = 1000,
  topic_labels = NULL,
  colors = NULL,
  verbose = TRUE,
  ...
)
```

## Arguments

- stm_model:

  A fitted STM model object.

- gamma_data:

  Optional pre-computed gamma data frame (default: NULL).

- top_n:

  The number of topics to display (default: 10).

- height:

  The height of the resulting Plotly plot, in pixels (default: 800).

- width:

  The width of the resulting Plotly plot, in pixels (default: 1000).

- topic_labels:

  Optional topic labels (default: NULL).

- colors:

  Optional color palette for topics (default: NULL).

- verbose:

  Logical, if TRUE, prints progress messages.

- ...:

  Further arguments passed to tidytext::tidy.

## Value

A plotly object showing a bar plot of topic prevalence.

## See also

Other topic_modeling:
[`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md),
[`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md),
[`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md)
