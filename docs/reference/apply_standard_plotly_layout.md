# Apply Standard Plotly Layout

Applies consistent layout styling to plotly plots following
TextAnalysisR design standards. This ensures all plots have uniform
fonts, colors, margins, and interactive features.

## Usage

``` r
apply_standard_plotly_layout(
  plot,
  title = NULL,
  xaxis_title = NULL,
  yaxis_title = NULL,
  margin = list(t = 60, b = 80, l = 80, r = 40),
  show_legend = FALSE
)
```

## Arguments

- plot:

  A plotly plot object

- title:

  Plot title text (optional)

- xaxis_title:

  X-axis title (optional)

- yaxis_title:

  Y-axis title (optional)

- margin:

  List of margins: list(t, b, l, r) in pixels (default: list(t = 60, b =
  80, l = 80, r = 40))

- show_legend:

  Logical, whether to show legend (default: FALSE)

## Value

A plotly plot object with standardized layout

## Details

Design standards applied:

- Title: 18px Roboto, \#0c1f4a

- Axis titles: 16px Roboto, \#0c1f4a

- Axis tick labels: 16px Roboto, \#3B3B3B

- Hover tooltips: 16px Roboto

- WCAG AA compliant colors

## See also

Other visualization:
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
[`plot_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_dispersion.md),
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
library(plotly)
p <- plot_ly(x = 1:10, y = rnorm(10), type = "scatter", mode = "markers")
p %>% apply_standard_plotly_layout(
  title = "My Plot",
  xaxis_title = "X Values",
  yaxis_title = "Y Values"
)
} # }
```
