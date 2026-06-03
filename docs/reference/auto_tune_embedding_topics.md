# Auto-tune BERTopic Hyperparameters

Automatically searches for optimal hyperparameters for embedding-based
topic modeling. Evaluates multiple configurations of UMAP and HDBSCAN
parameters and returns the best model based on the specified metric.
Embeddings are generated once and reused across all configurations for
efficiency.

## Usage

``` r
auto_tune_embedding_topics(
  texts,
  embeddings = NULL,
  embedding_model = "all-MiniLM-L6-v2",
  n_trials = 12,
  metric = "silhouette",
  seed = 123,
  verbose = TRUE
)
```

## Arguments

- texts:

  Character vector of documents to analyze.

- embeddings:

  Precomputed embeddings matrix (optional). If NULL, embeddings are
  generated.

- embedding_model:

  Embedding model name (default: "all-MiniLM-L6-v2").

- n_trials:

  Maximum number of configurations to try (default: 12).

- metric:

  Optimization metric: "silhouette", "coherence", or "combined"
  (default: "silhouette").

- seed:

  Random seed for reproducibility.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing:

- best_config: Data frame with the optimal hyperparameter configuration

- best_model: The topic model fitted with optimal parameters

- all_results: List of all evaluated configurations with metrics

- n_trials_completed: Number of configurations successfully evaluated

## Details

The function searches over these parameters:

- n_neighbors: UMAP neighborhood size (5, 10, 15, 25)

- min_cluster_size: HDBSCAN minimum cluster size (3, 5, 10)

- cluster_selection_method: "eom" (broader) or "leaf" (finer-grained)

## Examples

``` r
if (interactive()) {
  texts <- c("Machine learning for image recognition",
             "Deep learning neural networks",
             "Natural language processing models",
             "Computer vision applications")

  tuning_result <- auto_tune_embedding_topics(
    texts = texts,
    n_trials = 6,
    metric = "silhouette",
    verbose = TRUE
  )

  # View best configuration
  tuning_result$best_config

  # Use the best model
  best_model <- tuning_result$best_model
}
```
