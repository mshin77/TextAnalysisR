# Compute Word Co-occurrence Network

Computes word co-occurrence networks with community detection and
network metrics. Supports multiple feature spaces: unigrams, n-grams,
and embeddings. Based on proven implementation for intuitive network
visualization.

## Usage

``` r
semantic_cooccurrence_network(
  dfm_object,
  doc_var = NULL,
  co_occur_n = 10,
  top_node_n = 30,
  node_label_size = 14,
  pattern = NULL,
  showlegend = TRUE,
  seed = NULL,
  feature_type = "words",
  ngram_range = 2,
  texts = NULL,
  embeddings = NULL
)
```

## Arguments

- dfm_object:

  A quanteda document-feature matrix (dfm).

- doc_var:

  A document-level metadata variable for categories (default: NULL).

- co_occur_n:

  Minimum co-occurrence count (default: 10).

- top_node_n:

  Number of top nodes to display based on degree centrality (default:
  30).

- node_label_size:

  Font size for node labels (default: 14).

- pattern:

  Regex pattern to filter specific words (default: NULL).

- showlegend:

  Whether to show community legend (default: TRUE).

- seed:

  Random seed for reproducible layout (default: NULL).

- feature_type:

  Feature space: "words", "ngrams", or "embeddings" (default: "words").

- ngram_range:

  N-gram size when feature_type = "ngrams" (default: 2).

- texts:

  Optional character vector of texts for n-gram creation (default:
  NULL).

- embeddings:

  Optional embedding matrix for embedding-based networks (default:
  NULL).

## Value

A list containing plot, table, nodes, edges, and stats

## See also

Other network:
[`semantic_correlation_network()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_correlation_network.md)
