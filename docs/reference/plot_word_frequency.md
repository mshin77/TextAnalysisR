# Plot Word Frequency

Creates a bar plot showing the most frequent words in a document-feature
matrix (dfm).

## Usage

``` r
plot_word_frequency(dfm_object, n = 20, height = NULL, width = NULL, ...)
```

## Arguments

- dfm_object:

  A document-feature matrix created by quanteda::dfm().

- n:

  The number of top words to display (default: 20).

- height:

  Plot height in pixels (default: 800). Kept for backward compatibility.

- width:

  Plot width in pixels (default: 1000). Kept for backward compatibility.

- ...:

  Additional arguments (kept for backward compatibility).

## Value

A ggplot object showing word frequency.

## See also

Other visualization:
[`create_standard_ggplot_theme()`](https://mshin77.github.io/TextAnalysisR/reference/create_standard_ggplot_theme.md),
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
[`plot_weighted_log_odds()`](https://mshin77.github.io/TextAnalysisR/reference/plot_weighted_log_odds.md)

## Examples

``` r
if (interactive()) {
  data(SpecialEduTech, package = "TextAnalysisR")
  texts <- SpecialEduTech$abstract[1:10]
  dfm <- quanteda::dfm(quanteda::tokens(texts))
  plot <- plot_word_frequency(dfm, n = 10)
  print(plot)
}
```
