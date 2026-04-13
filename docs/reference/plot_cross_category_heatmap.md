# Plot Cross-Category Similarity Comparison

Creates a faceted ggplot heatmap for cross-category document similarity
comparison. Accepts either a pre-built long-format data frame or
extracts from a similarity matrix.

## Usage

``` r
plot_cross_category_heatmap(
  similarity_data,
  docs_data = NULL,
  row_var = "ld_doc_name",
  col_var = "other_doc_name",
  value_var = "cosine_similarity",
  category_var = "other_category",
  row_category = NULL,
  col_categories = NULL,
  row_display_var = NULL,
  col_display_var = NULL,
  method_name = "Cosine",
  title = NULL,
  show_values = TRUE,
  row_label = "Documents",
  label_max_chars = 25,
  order_by_numeric = TRUE,
  height = 600,
  width = NULL
)
```

## Arguments

- similarity_data:

  Either a similarity matrix (square numeric matrix) or a data frame in
  long format with columns for row labels, column labels, similarity
  values, and category.

- docs_data:

  Data frame with document metadata (required if similarity_data is a
  matrix)

- row_var:

  Column name for row document labels (default: "ld_doc_name")

- col_var:

  Column name for column document labels (default: "other_doc_name")

- value_var:

  Column name for similarity values (default: "cosine_similarity")

- category_var:

  Column name for category in long-format data or docs_data (default:
  "other_category")

- row_category:

  Category for row documents (used with matrix input)

- col_categories:

  Categories for column documents (used with matrix input)

- row_display_var:

  Column name for row display labels in tooltip (default: NULL, uses
  row_var)

- col_display_var:

  Column name for column display labels in tooltip (default: NULL, uses
  col_var)

- method_name:

  Similarity method name for legend (default: "Cosine")

- title:

  Plot title (default: NULL)

- show_values:

  Logical; show similarity values as text on tiles (default: TRUE)

- row_label:

  Label for y-axis (default: "Documents")

- label_max_chars:

  Maximum characters for axis labels before truncation (default: 25)

- order_by_numeric:

  Logical; order by numeric ID extracted from labels (default: TRUE)

- height:

  Plot height (default: 600)

- width:

  Plot width (default: NULL)

## Value

A ggplot object

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
[`plot_entity_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_entity_frequencies.md),
[`plot_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_dispersion.md),
[`plot_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/plot_log_odds_ratio.md),
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
# With pre-built long-format data
plot_cross_category_heatmap(
  similarity_data = ld_similarities,
  row_var = "ld_doc_name",
  col_var = "other_doc_name",
  value_var = "cosine_similarity",
  category_var = "other_category",
  row_label = "SLD Documents"
)
} # }
```
