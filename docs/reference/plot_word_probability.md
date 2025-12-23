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
  ...
)
```

## Arguments

- top_topic_terms:

  A data frame containing topic terms with columns: topic, term, and
  beta. Typically created using get_topic_terms() or similar functions.

- topic_label:

  Optional topic labels. Can be either a named vector mapping topic
  numbers to labels, or a character string specifying a column name in
  top_topic_terms (default: NULL).

- ncol:

  Number of columns for facet wrap layout (default: 3).

- height:

  The height of the resulting Plotly plot, in pixels (default: 1200).

- width:

  The width of the resulting Plotly plot, in pixels (default: 800).

- ylab:

  Y-axis label (default: "Word probability").

- title:

  Plot title (default: NULL for auto-generated title).

- colors:

  Color palette for topics (default: NULL for auto-generated colors).

- measure_label:

  Label for the probability measure (default: "Beta").

- ...:

  Additional arguments passed to plotly::ggplotly().

## Value

A plotly object showing word probabilities faceted by topic.

## See also

Other visualization:
[`apply_standard_plotly_layout()`](https://mshin77.github.io/TextAnalysisR/reference/apply_standard_plotly_layout.md),
[`create_empty_plot_message()`](https://mshin77.github.io/TextAnalysisR/reference/create_empty_plot_message.md),
[`create_message_table()`](https://mshin77.github.io/TextAnalysisR/reference/create_message_table.md),
[`create_standard_ggplot_theme()`](https://mshin77.github.io/TextAnalysisR/reference/create_standard_ggplot_theme.md),
[`get_dt_options()`](https://mshin77.github.io/TextAnalysisR/reference/get_dt_options.md),
[`get_plotly_hover_config()`](https://mshin77.github.io/TextAnalysisR/reference/get_plotly_hover_config.md),
[`get_sentiment_color()`](https://mshin77.github.io/TextAnalysisR/reference/get_sentiment_color.md),
[`get_sentiment_colors()`](https://mshin77.github.io/TextAnalysisR/reference/get_sentiment_colors.md),
[`plot_cluster_terms()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cluster_terms.md),
[`plot_cross_category_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cross_category_heatmap.md),
[`plot_entity_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_entity_frequencies.md),
[`plot_mwe_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_mwe_frequency.md),
[`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md),
[`plot_pos_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_pos_frequencies.md),
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md),
[`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md),
[`plot_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_probability.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)

## Examples

``` r
if (interactive()) {
  top_terms <- data.frame(
    topic = rep(1:2, each = 5),
    term = c("learning", "student", "education", "school", "teacher",
             "technology", "computer", "digital", "software", "system"),
    beta = c(0.05, 0.04, 0.03, 0.02, 0.01, 0.06, 0.05, 0.04, 0.03, 0.02)
  )
  plot <- plot_word_probability(top_terms)
  print(plot)
}
```
