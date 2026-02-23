# Plot Per-Document Per-Topic Probabilities

Generates a bar plot showing the prevalence of each topic across all
documents.

## Usage

``` r
plot_topic_probability(
  gamma_data,
  top_n = 10,
  topic_labels = NULL,
  colors = NULL,
  ylab = "Topic Proportion",
  base_font_size = 14
)
```

## Arguments

- gamma_data:

  A data frame with gamma values from calculate_topic_probability().

- top_n:

  The number of topics to display (default: 10).

- topic_labels:

  Optional topic labels (default: NULL).

- colors:

  Optional color palette for topics (default: NULL).

- ylab:

  Y-axis label (default: "Topic Proportion").

- base_font_size:

  Base font size in pixels for the plot theme (default: 14). Axis text
  and strip text will be base_font_size + 2.

## Value

A ggplot2 object showing a bar plot of topic prevalence.

## See also

Other topic_modeling:
[`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md),
[`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md),
[`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md)
