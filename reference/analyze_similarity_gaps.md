# Analyze Similarity Gaps Between Categories

Identifies unique items, missing content, and cross-category learning
opportunities based on similarity thresholds. Useful for gap analysis in
policy documents, topic comparisons, or any cross-category similarity
study.

## Usage

``` r
analyze_similarity_gaps(
  similarity_data,
  ref_var = "ref_id",
  other_var = "other_id",
  similarity_var = "similarity",
  category_var = "other_category",
  ref_label_var = NULL,
  other_label_var = NULL,
  unique_threshold = 0.6,
  cross_policy_min = 0.6,
  cross_policy_max = 0.8
)
```

## Arguments

- similarity_data:

  A data frame with cross-category similarities, containing:

  ref_var

  :   Reference item identifier

  other_var

  :   Comparison item identifier

  similarity_var

  :   Similarity score

  category_var

  :   Category of comparison item

- ref_var:

  Name of column with reference item IDs (default: "ref_id").

- other_var:

  Name of column with comparison item IDs (default: "other_id").

- similarity_var:

  Name of column with similarity values (default: "similarity").

- category_var:

  Name of column with category information (default: "other_category").

- ref_label_var:

  Optional column with reference item labels (for output).

- other_label_var:

  Optional column with comparison item labels (for output).

- unique_threshold:

  Threshold below which reference items are considered unique (default:
  0.6).

- cross_policy_min:

  Minimum similarity for cross-policy opportunities (default: 0.6).

- cross_policy_max:

  Maximum similarity for cross-policy opportunities (default: 0.8).

## Value

A list containing:

- unique_items:

  Data frame of reference items with low similarity (unique content)

- missing_items:

  Data frame of comparison items with low similarity (content gaps)

- cross_policy:

  Data frame of items with moderate similarity (learning opportunities)

- summary_stats:

  Summary statistics by category

## See also

Other semantic:
[`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md),
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
[`validate_cross_models()`](https://mshin77.github.io/TextAnalysisR/reference/validate_cross_models.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# After extracting cross-category similarities
gap_analysis <- analyze_similarity_gaps(
  similarity_data = cross_sims,
  ref_var = "ref_id",
  other_var = "other_id",
  similarity_var = "similarity",
  category_var = "other_category",
  unique_threshold = 0.6
)

print(gap_analysis$unique_items)
print(gap_analysis$missing_items)
print(gap_analysis$summary_stats)
} # }
```
