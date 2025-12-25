# Plot Multi-Word Expression Frequency

Creates a bar plot showing multi-word expression frequencies with
optional source-based coloring to distinguish between detected and
manually added expressions.

## Usage

``` r
plot_mwe_frequency(
  mwe_data,
  title = "Multi-Word Expression Frequency",
  color_by_source = TRUE,
  primary_color = "#10B981",
  secondary_color = "#A855F7",
  height = 500,
  width = NULL
)
```

## Arguments

- mwe_data:

  Data frame containing MWE data with columns:

  - `feature`: The multi-word expression text

  - `frequency`: Frequency count

  - `rank`: (optional) Rank of the expression

  - `docfreq`: (optional) Document frequency

  - `source`: (optional) Source category (e.g., "Top 20", "Manual")

- title:

  Plot title (default: "Multi-Word Expression Frequency")

- color_by_source:

  Whether to color bars by source column (default: TRUE)

- primary_color:

  Color for primary/top expressions (default: "#10B981")

- secondary_color:

  Color for secondary/manual expressions (default: "#A855F7")

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
[`plot_entity_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_entity_frequencies.md),
[`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md),
[`plot_pos_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_pos_frequencies.md),
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)
