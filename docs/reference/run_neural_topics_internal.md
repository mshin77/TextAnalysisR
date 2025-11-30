# Neural Topic Modeling

Implements neural topic modeling using deep learning architectures for
improved topic discovery and representation learning.

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

  Number of hidden layers in neural network

- hidden_units:

  Number of units per hidden layer

- dropout_rate:

  Dropout rate for regularization

- embedding_model:

  Transformer model for initial embeddings

- seed:

  Random seed for reproducibility

## Value

List containing neural topic model and diagnostics
