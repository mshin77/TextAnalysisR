# Analyze and Visualize Word Co-occurrence Networks

This function creates a word co-occurrence network based on a
document-feature matrix (dfm).

## Usage

``` r
word_co_occurrence_network(
  dfm_object,
  doc_var = NULL,
  co_occur_n = 50,
  top_node_n = 30,
  nrows = 1,
  height = 800,
  width = 900,
  node_label_size = 22,
  community_method = "leiden",
  node_size_by = "degree",
  node_color_by = "community"
)
```

## Arguments

- dfm_object:

  A quanteda document-feature matrix (dfm).

- doc_var:

  A document-level metadata variable (default: NULL).

- co_occur_n:

  Minimum co-occurrence count (default: 50).

- top_node_n:

  Number of top nodes to display (default: 30).

- nrows:

  Number of rows to display in the table (default: 1).

- height:

  The height of the resulting Plotly plot, in pixels (default: 800).

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

## Value

A list containing the Plotly plot, a table, and a summary.

## See also

Other semantic:
[`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md),
[`analyze_similarity_gaps()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md),
[`calculate_clustering_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_clustering_metrics.md),
[`calculate_cross_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_cross_similarity.md),
[`calculate_document_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_document_similarity.md),
[`calculate_similarity_robust()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_similarity_robust.md),
[`cluster_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/cluster_embeddings.md),
[`cross_analysis_validation()`](https://mshin77.github.io/TextAnalysisR/reference/cross_analysis_validation.md),
[`export_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/export_document_clustering.md),
[`extract_cross_category_similarities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_cross_category_similarities.md),
[`fit_semantic_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_semantic_model.md),
[`generate_cluster_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels.md),
[`generate_cluster_labels_auto()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels_auto.md),
[`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md),
[`reduce_dimensions()`](https://mshin77.github.io/TextAnalysisR/reference/reduce_dimensions.md),
[`semantic_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_document_clustering.md),
[`semantic_similarity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_similarity_analysis.md),
[`temporal_semantic_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/temporal_semantic_analysis.md),
[`validate_cross_models()`](https://mshin77.github.io/TextAnalysisR/reference/validate_cross_models.md),
[`word_correlation_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_correlation_network.md)

## Examples

``` r
if (interactive()) {
  df <- TextAnalysisR::SpecialEduTech

  united_tbl <- TextAnalysisR::unite_cols(df, listed_vars = c("title", "abstract"))

  tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")

  dfm_object <- quanteda::dfm(tokens)

  word_co_occurrence_network_results <- TextAnalysisR::word_co_occurrence_network(
                                        dfm_object,
                                        doc_var = NULL,
                                        co_occur_n = 50,
                                        top_node_n = 30,
                                        nrows = 1,
                                        height = 800,
                                        width = 900,
                                        community_method = "leiden")
  print(word_co_occurrence_network_results$plot)
  print(word_co_occurrence_network_results$table)
  print(word_co_occurrence_network_results$summary)
}
```
