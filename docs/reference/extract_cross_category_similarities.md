# Extract Cross-Category Similarities from Full Similarity Matrix

Given a full similarity matrix and category information, extracts
pairwise similarities between a reference category and other categories
into a long-format data frame suitable for visualization and analysis.

## Usage

``` r
extract_cross_category_similarities(
  similarity_matrix,
  docs_data,
  reference_category,
  compare_categories = NULL,
  category_var = "category",
  id_var = "display_name",
  name_var = NULL
)
```

## Arguments

- similarity_matrix:

  A square similarity matrix (n x n).

- docs_data:

  A data frame containing document metadata with at least:

  category_var

  :   Column indicating category membership

  id_var

  :   Column with unique document identifiers

- reference_category:

  Character string specifying the reference category to compare against.

- compare_categories:

  Character vector of categories to compare with the reference. If NULL,
  compares with all categories except reference.

- category_var:

  Name of the column containing category information (default:
  "category").

- id_var:

  Name of the column containing document IDs (default: "display_name").

- name_var:

  Optional name of column with display names (default: NULL, uses
  id_var).

## Value

A data frame with columns:

- ref_id:

  Reference document ID

- ref_name:

  Reference document name (if name_var provided)

- other_id:

  Comparison document ID

- other_name:

  Comparison document name (if name_var provided)

- other_category:

  Category of comparison document

- similarity:

  Cosine similarity value

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
[`fit_semantic_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_semantic_model.md),
[`generate_cluster_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels.md),
[`generate_cluster_labels_auto()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels_auto.md),
[`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md),
[`reduce_dimensions()`](https://mshin77.github.io/TextAnalysisR/reference/reduce_dimensions.md),
[`semantic_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_document_clustering.md),
[`semantic_similarity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_similarity_analysis.md),
[`temporal_semantic_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/temporal_semantic_analysis.md),
[`validate_cross_models()`](https://mshin77.github.io/TextAnalysisR/reference/validate_cross_models.md),
[`word_co_occurrence_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_co_occurrence_network.md),
[`word_correlation_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_correlation_network.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# After calculating full similarity matrix
similarity_result <- TextAnalysisR::calculate_document_similarity(
  texts = docs$text,
  document_feature_type = "embeddings"
)

cross_sims <- extract_cross_category_similarities(
  similarity_matrix = similarity_result$similarity_matrix,
  docs_data = docs,
  reference_category = "SLD",
  compare_categories = c("Other Disability", "General"),
  category_var = "category",
  id_var = "display_name",
  name_var = "doc_name"
)
} # }
```
