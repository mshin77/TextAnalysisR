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

## Examples

``` r
if (interactive()) {
  texts <- c(
    "Assistive technology supports learning for students with disabilities.",
    "Technology aids help disabled students with their education.",
    "Machine learning algorithms improve predictive accuracy."
  )

  result <- calculate_document_similarity(
    texts = texts,
    document_feature_type = "words",
    similarity_method = "cosine"
  )

  print(result$similarity_matrix)
  print(result$metrics)
}
```
