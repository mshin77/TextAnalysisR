# Plot Part-of-Speech Tag Frequencies

Creates a bar plot showing the frequency distribution of part-of-speech
tags.

## Usage

``` r
plot_pos_frequencies(
  pos_data,
  top_n = 20,
  title = "Part-of-Speech Tag Frequency",
  color = "#337ab7",
  height = 500,
  width = NULL
)
```

## Arguments

- pos_data:

  Data frame containing POS data with columns:

  - `pos`: Part-of-speech tag

  - `n`: (optional) Pre-computed frequency count

  If `n` is not present, frequencies will be computed from the data.

- top_n:

  Number of top POS tags to display (default: 20)

- title:

  Plot title (default: "Part-of-Speech Tag Frequency")

- color:

  Bar color (default: "#337ab7")

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
[`plot_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_dispersion.md),
[`plot_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/plot_log_odds_ratio.md),
[`plot_mwe_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_mwe_frequency.md),
[`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md),
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)

## Examples

``` r
if (interactive()) {
  pos_df <- data.frame(
    pos = c("NOUN", "VERB", "ADJ", "ADV", "PRON"),
    n = c(500, 400, 250, 150, 100)
  )
  plot_pos_frequencies(pos_df)
}
```
