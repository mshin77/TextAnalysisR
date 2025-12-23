# Plot Named Entity Frequencies

Creates a bar plot showing the frequency distribution of named entity
types.

## Usage

``` r
plot_entity_frequencies(
  entity_data,
  top_n = 20,
  title = "Named Entity Type Frequency",
  color = "#10B981",
  height = 500,
  width = NULL
)
```

## Arguments

- entity_data:

  Data frame containing entity data with columns:

  - `entity`: Named entity type (e.g., "PERSON", "ORG", "GPE")

  - `n`: (optional) Pre-computed frequency count

  If `n` is not present, frequencies will be computed from the data.

- top_n:

  Number of top entity types to display (default: 20)

- title:

  Plot title (default: "Named Entity Type Frequency")

- color:

  Bar color (default: "#10B981")

- height:

  Plot height in pixels (default: 500)

- width:

  Plot width in pixels (default: NULL for auto)

## Value

A plotly object

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
[`plot_mwe_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_mwe_frequency.md),
[`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md),
[`plot_pos_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_pos_frequencies.md),
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md),
[`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md),
[`plot_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_probability.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md),
[`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md)

## Examples

``` r
if (interactive()) {
  entity_df <- data.frame(
    entity = c("PERSON", "ORG", "GPE", "DATE", "MONEY"),
    n = c(300, 250, 200, 150, 100)
  )
  plot_entity_frequencies(entity_df)
}
```
