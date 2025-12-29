# Calculate Document Similarity

Calculates similarity between documents using traditional NLP methods or
modern embedding-based approaches. Comprehensive metrics are
automatically computed unless disabled.

## Usage

``` r
calculate_document_similarity(
  texts,
  document_feature_type = "words",
  semantic_ngram_range = 2,
  similarity_method = "cosine",
  use_embeddings = FALSE,
  embedding_model = "all-MiniLM-L6-v2",
  calculate_metrics = TRUE,
  verbose = TRUE
)
```

## Arguments

- texts:

  A character vector of texts to compare.

- document_feature_type:

  Feature extraction type: "words", "ngrams", "embeddings", or "topics".

- semantic_ngram_range:

  Integer, n-gram range for ngram features (default: 2).

- similarity_method:

  Similarity calculation method: "cosine", "jaccard", "euclidean",
  "manhattan".

- use_embeddings:

  Logical, use embedding-based similarity (default: FALSE).

- embedding_model:

  Sentence transformer model name (default: "all-MiniLM-L6-v2").

- calculate_metrics:

  Logical, compute comprehensive similarity metrics (default: TRUE).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing:

- similarity_matrix:

  N x N similarity matrix

- feature_matrix:

  Document feature matrix used for calculation

- method_info:

  Information about the method used

- metrics:

  Comprehensive similarity metrics (if calculate_metrics = TRUE)

- execution_time:

  Time taken for analysis

## See also

Other semantic:
[`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md),
[`analyze_similarity_gaps()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md),
[`calculate_clustering_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_clustering_metrics.md),
[`calculate_cross_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_cross_similarity.md),
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
[`word_co_occurrence_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_co_occurrence_network.md),
[`word_correlation_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_correlation_network.md)

## Examples

``` r
if (interactive()) {
  data(SpecialEduTech)
  texts <- SpecialEduTech$abstract[1:5]

  result <- calculate_document_similarity(
    texts = texts,
    document_feature_type = "words",
    similarity_method = "cosine"
  )

  print(result$similarity_matrix)
  print(result$metrics)
}
```
