# Plot Per-Document Per-Topic Probabilities

Generates a bar plot showing the prevalence of each topic across all
documents.

## Usage

``` r
plot_topic_probability(
  gamma_data,
  top_n = 10,
  use_topic_labels = FALSE,
  colors = NULL,
  ylab = "Topic Proportion",
  base_font_size = 11
)
```

## Arguments

- gamma_data:

  A data frame with gamma values from calculate_topic_probability().

- top_n:

  The number of topics to display (default: 10).

- use_topic_labels:

  Logical. If TRUE, use the `topic_label` column from `gamma_data` for
  axis labels (falls back to topic number when the column is absent). If
  FALSE (default), labels are formatted as "Topic N".

- colors:

  Optional color palette for topics (default: NULL).

- ylab:

  Y-axis label (default: "Topic Proportion").

- base_font_size:

  Base font size in points for the plot theme (default: 11). Axis text
  and strip text will be base_font_size + 2.

## Value

A ggplot2 object showing a bar plot of topic prevalence.
