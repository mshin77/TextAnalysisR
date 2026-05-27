# Analyze and Visualize Word Correlation Networks

This function creates a word correlation network based on a
document-feature matrix (dfm).

## Usage

``` r
word_correlation_network(
  dfm_object,
  doc_var = NULL,
  common_term_n = 130,
  corr_n = 0.4,
  top_node_n = 40,
  nrows = 1,
  height = 1000,
  width = 900,
  node_label_size = 22,
  community_method = "leiden",
  node_size_by = "degree",
  node_color_by = "community",
  category_params = NULL,
  pattern = NULL,
  seed = 2025,
  physics_gravity = -1500,
  physics_spring_length = 100,
  physics_avoid_overlap = 0.3,
  showlegend = TRUE
)
```

## Arguments

- dfm_object:

  A quanteda document-feature matrix (dfm).

- doc_var:

  A document-level metadata variable (default: NULL).

- common_term_n:

  Minimum number of common terms for filtering terms (default: 130).

- corr_n:

  Minimum correlation value for filtering terms (default: 0.4).

- top_node_n:

  Number of top nodes to display (default: 40).

- nrows:

  Number of rows to display in the table (default: 1).

- height:

  The height of the resulting Plotly plot, in pixels (default: 1000).

- width:

  The width of the resulting Plotly plot, in pixels (default: 900).

- node_label_size:

  Maximum font size for node labels in pixels (default: 22).

- community_method:

  Community detection method: "leiden" (default) or "louvain".

- node_size_by:

  Node sizing method: "degree", "betweenness", "closeness",
  "eigenvector", or "fixed" (default: "degree").

- node_color_by:

  Node coloring method: "community" or "centrality" (default:
  "community").

- category_params:

  Optional named list of category-specific parameters. Each element
  should be a list with `common_term_n`, `corr_n`, and `top_node_n`
  values for that category (default: NULL).

- pattern:

  Optional regex (case-insensitive). If provided, keeps only edges whose
  `item1` or `item2` matches (default: NULL).

- seed:

  Integer RNG seed for reproducible layout (default: 2025).

- physics_gravity:

  barnesHut `gravitationalConstant`. More negative = nodes spread
  further apart (default: -1500).

- physics_spring_length:

  barnesHut spring length. Higher = longer edges (default: 100).

- physics_avoid_overlap:

  barnesHut overlap avoidance, 0 to 1. Higher = more node separation
  (default: 0.3).

- showlegend:

  Whether to display the community legend (default: TRUE).

## Value

A list containing the visNetwork widget (or flex-layout HTML for
faceted), a DT table, and a summary.

## Examples

``` r
if (interactive()) {
  df <- TextAnalysisR::SpecialEduTech

  united_tbl <- TextAnalysisR::unite_cols(df, listed_vars = c("title", "abstract"))

  tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")

  dfm_object <- quanteda::dfm(tokens)

  word_correlation_network_results <- TextAnalysisR::word_correlation_network(
                                      dfm_object,
                                      doc_var = NULL,
                                      common_term_n = 30,
                                      corr_n = 0.4,
                                      top_node_n = 40,
                                      nrows = 1,
                                      height = 1000,
                                      width = 900,
                                      community_method = "leiden")
  print(word_correlation_network_results$plot)
  print(word_correlation_network_results$table)
  print(word_correlation_network_results$summary)
}
```
