# Calculate Semantic Topic Quality Metrics

Internal function to calculate quality metrics for semantic topic
modeling results.

## Usage

``` r
calculate_topic_quality(
  embeddings,
  topic_assignments,
  similarity_matrix = NULL
)
```

## Arguments

- embeddings:

  Document embeddings matrix.

- topic_assignments:

  Vector of topic assignments.

- similarity_matrix:

  Optional similarity matrix.

## Value

A list of quality metrics.
