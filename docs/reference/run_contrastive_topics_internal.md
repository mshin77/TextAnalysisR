# Contrastive Learning Topic Modeling

Implements contrastive learning approaches for topic modeling to improve
topic separation and discriminability.

## Usage

``` r
run_contrastive_topics_internal(
  texts,
  n_topics = 10,
  temperature = 0.1,
  negative_sampling_rate = 5,
  embedding_model = "all-MiniLM-L6-v2",
  seed = 123
)
```

## Arguments

- texts:

  Character vector of documents

- n_topics:

  Number of topics to discover

- temperature:

  Temperature parameter for contrastive learning

- negative_sampling_rate:

  Rate of negative sampling

- embedding_model:

  Transformer model for embeddings

- seed:

  Random seed for reproducibility

## Value

List containing contrastive topic model and metrics
