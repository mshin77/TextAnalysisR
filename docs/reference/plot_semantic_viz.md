# Plot Semantic Analysis Visualization

Creates interactive visualizations for semantic analysis results
including similarity heatmaps, dimensionality reduction plots, and
clustering visualizations.

## Usage

``` r
plot_semantic_viz(
  analysis_result = NULL,
  plot_type = "similarity",
  data_labels = NULL,
  color_by = NULL,
  height = 600,
  width = 800,
  title = NULL,
  coords = NULL,
  clusters = NULL,
  hover_text = NULL,
  hover_config = NULL,
  cluster_colors = NULL
)
```

## Arguments

- analysis_result:

  A list containing semantic analysis results from functions like
  semantic_similarity_analysis(), semantic_document_clustering(), or
  reduce_dimensions().

- plot_type:

  Type of visualization: "similarity" for heatmap,
  "dimensionality_reduction" for scatter plot, or "clustering" for
  cluster visualization (default: "similarity").

- data_labels:

  Optional character vector of labels for data points (default: NULL).

- color_by:

  Optional variable to color points by in scatter plots (default: NULL).

- height:

  The height of the resulting Plotly plot, in pixels (default: 600).

- width:

  The width of the resulting Plotly plot, in pixels (default: 800).

- title:

  Optional custom title for the plot (default: NULL).

- coords:

  Optional pre-computed coordinates for dimensionality reduction plots
  (default: NULL).

- clusters:

  Optional cluster assignments vector (default: NULL).

- hover_text:

  Optional custom hover text for points (default: NULL).

- hover_config:

  Optional hover configuration list (default: NULL).

- cluster_colors:

  Optional color palette for clusters (default: NULL).

## Value

A plotly object showing the specified visualization.

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
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)

## Examples

``` r
if (interactive()) {
  texts <- c("machine learning", "deep learning", "artificial intelligence")
  result <- semantic_similarity_analysis(texts)
  plot <- plot_semantic_viz(result, plot_type = "similarity")
  print(plot)
}
```
