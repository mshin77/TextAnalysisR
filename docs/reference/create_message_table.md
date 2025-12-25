# Create Message Data Table

Creates a formatted DT::datatable displaying an informational message.
Useful for showing status messages in place of empty tables.

## Usage

``` r
create_message_table(message, font_size = "16px", color = "#6c757d")
```

## Arguments

- message:

  Character string message to display

- font_size:

  Font size (default: "16px")

- color:

  Text color (default: "#6c757d")

## Value

A DT::datatable object

## See also

Other visualization:
[`apply_standard_plotly_layout()`](https://mshin77.github.io/TextAnalysisR/reference/apply_standard_plotly_layout.md),
[`create_empty_plot_message()`](https://mshin77.github.io/TextAnalysisR/reference/create_empty_plot_message.md),
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
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)

## Examples

``` r
if (FALSE) { # \dontrun{
create_message_table("No data available. Please run analysis first.")
} # }
```
