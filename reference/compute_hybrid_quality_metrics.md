# Compute Hybrid Model Quality Metrics

Computes quality metrics for hybrid topic models including semantic
coherence, exclusivity, and silhouette scores. Based on research
recommendations for topic model evaluation (Roberts et al., Mimno et
al.).

## Usage

``` r
compute_hybrid_quality_metrics(
  stm_model,
  stm_documents,
  embedding_result,
  embeddings = NULL
)
```

## Arguments

- stm_model:

  A fitted STM model object.

- stm_documents:

  STM-formatted documents.

- embedding_result:

  Result from fit_embedding_topics().

- embeddings:

  Document embeddings matrix (optional, for silhouette).

## Value

A list containing quality metrics:

- stm_coherence: Semantic coherence per STM topic

- stm_exclusivity: Exclusivity per STM topic

- stm_coherence_mean: Mean semantic coherence

- stm_exclusivity_mean: Mean exclusivity

- embedding_silhouette: Silhouette scores for embedding clusters

- embedding_silhouette_mean: Mean silhouette score

- combined_quality: Overall quality score
