# Plot Term Frequency Trends by Continuous Variable

Creates a faceted line plot showing how term frequencies vary across a
continuous variable (e.g., year, time period).

## Usage

``` r
plot_term_trends_continuous(
  term_data,
  continuous_var,
  terms = NULL,
  title = NULL,
  height = 600,
  width = NULL
)
```

## Arguments

- term_data:

  Data frame containing term frequencies with columns: continuous_var,
  term, and word_frequency

- continuous_var:

  Name of the continuous variable column

- terms:

  Character vector of terms to display (optional, filters if provided)

- title:

  Plot title (default: NULL, auto-generated)

- height:

  Plot height in pixels (default: 600)

- width:

  Plot width in pixels (default: NULL, auto)

## Value

A plotly object with faceted line plots

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
[`plot_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/plot_log_odds_ratio.md),
[`plot_mwe_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_mwe_frequency.md),
[`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md),
[`plot_pos_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_pos_frequencies.md),
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_weighted_log_odds()`](https://mshin77.github.io/TextAnalysisR/reference/plot_weighted_log_odds.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)

## Examples

``` r
if (FALSE) { # \dontrun{
term_df <- data.frame(
  year = rep(2010:2020, each = 3),
  term = rep(c("learning", "education", "technology"), 11),
  word_frequency = sample(10:100, 33, replace = TRUE)
)
plot_term_trends_continuous(term_df, "year", c("learning", "education"))
} # }
```
