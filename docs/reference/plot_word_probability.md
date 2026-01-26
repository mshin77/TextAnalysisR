# Plot Word Probabilities by Topic

Creates a faceted bar plot showing the top terms and their probabilities
(beta values) for each topic in a topic model.

## Usage

``` r
plot_word_probability(
  top_topic_terms,
  topic_label = NULL,
  ncol = 3,
  height = 1200,
  width = 800,
  ylab = "Word probability",
  title = NULL,
  colors = NULL,
  measure_label = "Beta",
  base_font_size = 14,
  ...
)
```

## Arguments

- top_topic_terms:

  A data frame containing topic terms with columns: topic, term, and
  beta.

- topic_label:

  Optional topic labels. Can be either a named vector mapping topic
  numbers to labels, or a character string specifying a column name in
  top_topic_terms (default: NULL).

- ncol:

  Number of columns for facet wrap layout (default: 3).

- height:

  Plot height for responsive spacing adjustments (default: 1200).

- width:

  Plot width for responsive spacing adjustments (default: 800).

- ylab:

  Y-axis label (default: "Word probability").

- title:

  Plot title (default: NULL for auto-generated title).

- colors:

  Color palette for topics (default: NULL for auto-generated colors).

- measure_label:

  Label for the probability measure (default: "Beta").

- base_font_size:

  Base font size in pixels for the plot theme (default: 14). Axis text
  and strip text will be base_font_size + 2.

- ...:

  Additional arguments (currently unused, kept for compatibility).

## Value

A ggplot2 object showing word probabilities faceted by topic.

## See also

Other topic_modeling:
[`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md),
[`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md),
[`plot_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_probability.md)
