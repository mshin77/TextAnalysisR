# Calculate Document Similarity with Fallbacks

Calculates document similarity with fallback methods and diagnostics.
Attempts embeddings first, falls back to Jaccard similarity if needed.

## Usage

``` r
calculate_similarity_robust(
  texts,
  method = "embeddings",
  embedding_model = "all-MiniLM-L6-v2",
  cache_embeddings = TRUE,
  min_word_length = 3,
  doc_names = NULL
)
```

## Arguments

- texts:

  Character vector of texts

- method:

  Similarity method ("embeddings" or "jaccard")

- embedding_model:

  Model name for embeddings (default: "all-MiniLM-L6-v2")

- cache_embeddings:

  Logical, cache embeddings (default: TRUE)

- min_word_length:

  Minimum word length for Jaccard (default: 3)

- doc_names:

  Optional document names

## Value

List containing similarity matrix, method used, embeddings, and
diagnostics

## Examples

``` r
if (interactive()) {
  abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:5]
  similarity_result <- calculate_similarity_robust(abstracts)
  print(similarity_result$similarity_matrix)
  print(similarity_result$diagnostics)
}
```
