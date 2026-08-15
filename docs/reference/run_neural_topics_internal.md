# Neural Topic Modeling (deprecated)

Deprecated. Use
[`cluster_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/cluster_embedding_topics.md).
The `hidden_layers`, `hidden_units`, and `dropout_rate` arguments never
affected the result and are ignored.

## Usage

``` r
run_neural_topics_internal(
  texts,
  n_topics = 10,
  hidden_layers = 2,
  hidden_units = 100,
  dropout_rate = 0.2,
  embedding_model = "all-MiniLM-L6-v2",
  seed = 123
)
```

## Arguments

- texts:

  Character vector of documents

- n_topics:

  Number of topics to discover

- hidden_layers:

  Ignored.

- hidden_units:

  Ignored.

- dropout_rate:

  Ignored.

- embedding_model:

  Transformer model for initial embeddings

- seed:

  Random seed for reproducibility

## Value

See
[`cluster_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/cluster_embedding_topics.md).
