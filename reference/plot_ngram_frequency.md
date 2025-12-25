# Plot N-gram Frequency

Creates a bar plot showing n-gram frequencies with optional highlighting
of selected n-grams. Supports both detected n-grams and selected
multi-word expressions.

## Usage

``` r
plot_ngram_frequency(
  ngram_data,
  top_n = 30,
  selected = NULL,
  title = "N-gram Frequency",
  highlight_color = "#10B981",
  default_color = "#6B7280",
  height = 500,
  width = NULL,
  show_stats = TRUE
)
```

## Arguments

- ngram_data:

  Data frame containing n-gram data with columns:

  - `collocation`: The n-gram text

  - `count`: Frequency count

  - `lambda`: (optional) Lambda statistic

  - `z`: (optional) Z-score statistic

- top_n:

  Number of top n-grams to display (default: 30)

- selected:

  Character vector of selected n-grams to highlight (default: NULL)

- title:

  Plot title (default: "N-gram Frequency")

- highlight_color:

  Color for highlighted bars (default: "#10B981")

- default_color:

  Color for non-highlighted bars (default: "#6B7280")

- height:

  Plot height in pixels (default: 500)

- width:

  Plot width in pixels (default: NULL for auto)

- show_stats:

  Whether to show lambda and z-score in hover (default: TRUE)

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
[`plot_mwe_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_mwe_frequency.md),
[`plot_pos_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_pos_frequencies.md),
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)

## Examples

``` r
if (interactive()) {
  ngram_df <- data.frame(
    collocation = c("machine learning", "deep learning", "neural network"),
    count = c(150, 120, 90),
    lambda = c(5.2, 4.8, 4.1),
    z = c(12.3, 10.5, 9.2)
  )
  plot_ngram_frequency(ngram_df, selected = c("machine learning"))
}
```
