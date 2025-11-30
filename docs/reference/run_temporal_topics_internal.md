# Temporal Dynamic Topic Modeling

Analyzes topic evolution over time periods using dynamic modeling
approaches to track concept emergence, evolution, and decline.

## Usage

``` r
run_temporal_topics_internal(
  texts,
  metadata = NULL,
  n_topics = 10,
  temporal_unit = "year",
  temporal_window = 3,
  detect_evolution = TRUE,
  embedding_model = "all-MiniLM-L6-v2",
  seed = 123
)
```

## Arguments

- texts:

  Character vector of documents

- metadata:

  Data frame containing temporal information

- n_topics:

  Number of topics to discover

- temporal_unit:

  Unit for temporal analysis ("year", "quarter", "month")

- temporal_window:

  Size of temporal window for analysis

- detect_evolution:

  Whether to detect topic evolution patterns

- embedding_model:

  Transformer model for embeddings

- seed:

  Random seed for reproducibility

## Value

List containing temporal topic model and evolution analysis
