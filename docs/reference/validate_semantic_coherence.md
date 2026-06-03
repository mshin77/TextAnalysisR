# Validate Semantic Coherence

Validates the semantic coherence of topic assignments using
intra-cluster distance in embedding space.

## Usage

``` r
validate_semantic_coherence(embeddings, topic_assignments, ...)
```

## Arguments

- embeddings:

  Document embeddings matrix.

- topic_assignments:

  Vector of topic assignments for documents.

- ...:

  Additional parameters (currently unused).

## Value

List containing coherence score and metrics.
