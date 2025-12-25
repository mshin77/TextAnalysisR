# Plot Document Similarity Heatmap

Creates an interactive heatmap visualization of document similarity
matrices with support for document metadata, feature-specific
colorscales, and rich tooltips. Supports both symmetric (all-vs-all) and
cross-category comparison modes.

## Usage

``` r
plot_similarity_heatmap(
  similarity_matrix,
  docs_data = NULL,
  feature_type = "words",
  method_name = "Cosine",
  title = NULL,
  category_filter = NULL,
  doc_id_var = NULL,
  colorscale = NULL,
  height = 600,
  width = NULL,
  row_category = NULL,
  col_categories = NULL,
  category_var = "category_display",
  show_values = FALSE,
  facet = NULL,
  row_label = NULL,
  output_type = "plotly"
)
```

## Arguments

- similarity_matrix:

  A square numeric matrix of similarity scores

- docs_data:

  Optional data frame with document metadata containing:

  - `document_number`: Document identifiers for axis labels

  - `document_id_display`: Document IDs for hover text

  - `category_display`: Category labels for hover text

- feature_type:

  Feature space type: "words", "topics", "ngrams", or "embeddings"
  (determines colorscale and display name)

- method_name:

  Similarity method name for display (default: "Cosine")

- title:

  Plot title (default: NULL, auto-generated from feature_type)

- category_filter:

  Optional category filter label for title (default: NULL)

- doc_id_var:

  Name of document ID variable (affects label text, default: NULL)

- colorscale:

  Plotly colorscale override (default: NULL, uses feature_type default)

- height:

  Plot height in pixels (default: 600)

- width:

  Plot width in pixels (default: NULL for auto)

- row_category:

  Category for row documents in cross-category mode (default: NULL)

- col_categories:

  Character vector of categories for column documents (default: NULL)

- category_var:

  Name of category variable in docs_data (default: "category_display")

- show_values:

  Logical; show similarity values as text on tiles (default: FALSE)

- facet:

  Logical; facet by column categories (default: TRUE when col_categories
  specified)

- row_label:

  Label for row axis (default: NULL, uses row_category)

- output_type:

  Output type: "plotly" or "ggplot" (default: "plotly", auto-switches to
  "ggplot" for faceting)

## Value

A plotly or ggplot2 heatmap object

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
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Simple usage with matrix only
sim_matrix <- matrix(runif(25), nrow = 5)
plot_similarity_heatmap(sim_matrix)

# With document metadata
docs <- data.frame(
  document_number = paste("Doc", 1:5),
  document_id_display = c("Paper A", "Paper B", "Paper C", "Paper D", "Paper E"),
  category_display = c("Science", "Science", "Tech", "Tech", "Health")
)
plot_similarity_heatmap(sim_matrix, docs_data = docs, feature_type = "embeddings")

# Cross-category comparison with faceting
plot_similarity_heatmap(
  sim_matrix,
  docs_data = docs,
  row_category = "Science",
  col_categories = c("Tech", "Health"),
  show_values = TRUE,
  facet = TRUE
)
} # }
```
