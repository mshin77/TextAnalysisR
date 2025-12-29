# Plot Lexical Dispersion

Creates an X-ray plot showing where terms appear across documents. Each
row represents a term, and marks indicate occurrences.

## Usage

``` r
plot_lexical_dispersion(
  dispersion_data,
  scale = "relative",
  title = "Lexical Dispersion",
  colors = NULL,
  height = 400,
  width = NULL,
  marker_size = 8
)
```

## Arguments

- dispersion_data:

  Data frame from calculate_lexical_dispersion()

- scale:

  Character, "relative" or "absolute" (must match calculation)

- title:

  Plot title (default: "Lexical Dispersion")

- colors:

  Named vector of colors for each term, or NULL for auto

- height:

  Plot height in pixels (default: 400)

- width:

  Plot width in pixels (default: NULL for auto)

- marker_size:

  Size of position markers (default: 8)

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
[`plot_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/plot_log_odds_ratio.md),
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
dispersion <- calculate_lexical_dispersion(tokens, c("education", "technology"))
plot_lexical_dispersion(dispersion)
} # }
```
