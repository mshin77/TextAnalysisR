# Plot Log Odds Ratio

Creates a horizontal bar plot showing log odds ratios for comparing word
usage between categories. Positive values indicate higher usage in the
first category, negative in the second.

## Usage

``` r
plot_log_odds_ratio(
  log_odds_data,
  top_n = 10,
  facet_by = NULL,
  color_positive = "#10B981",
  color_negative = "#EF4444",
  height = 600,
  width = NULL,
  title = "Log Odds Ratio Comparison"
)
```

## Arguments

- log_odds_data:

  Data frame from calculate_log_odds_ratio()

- top_n:

  Number of top terms to show per direction (default: 10)

- facet_by:

  Character, column name to facet by (e.g., "category1" for one_vs_rest
  comparisons). NULL for no faceting.

- color_positive:

  Color for positive log odds (default: "#10B981" green)

- color_negative:

  Color for negative log odds (default: "#EF4444" red)

- height:

  Plot height in pixels (default: 600)

- width:

  Plot width in pixels (default: NULL for auto)

- title:

  Plot title (default: "Log Odds Ratio Comparison")

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
[`plot_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_dispersion.md),
[`plot_mwe_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_mwe_frequency.md),
[`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md),
[`plot_pos_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_pos_frequencies.md),
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_weighted_log_odds()`](https://mshin77.github.io/TextAnalysisR/reference/plot_weighted_log_odds.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)

## Examples

``` r
if (FALSE) { # \dontrun{
log_odds <- calculate_log_odds_ratio(dfm, "category", comparison_mode = "binary")
plot_log_odds_ratio(log_odds, top_n = 15)
} # }
```
