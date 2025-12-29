# Compute Topic Alignment Using Cosine Similarity

Computes alignment between STM and embedding-based topics using cosine
similarity on topic-word distributions and document-topic assignments.
This method follows research best practices for cross-model topic
alignment.

## Usage

``` r
calculate_topic_alignment(stm_model, embedding_result, stm_vocab, texts)
```

## Arguments

- stm_model:

  A fitted STM model object.

- embedding_result:

  Result from fit_embedding_model().

- stm_vocab:

  Vocabulary from STM conversion.

- texts:

  Original texts for computing embedding topic centroids.

## Value

A list containing alignment metrics:

- alignment_matrix: Cosine similarity matrix between topics

- best_matches: Best matching embedding topic for each STM topic

- alignment_scores: Alignment score per topic

- overall_alignment: Mean alignment across all topics

- assignment_agreement: Agreement between document assignments

- correlation: Correlation between assignment vectors
