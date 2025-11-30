# Embedding-based Topic Modeling

This function performs embedding-based topic modeling using transformer
embeddings and specialized clustering techniques. The primary method
uses the BERTopic library, which combines transformer embeddings with
UMAP dimensionality reduction and HDBSCAN clustering for optimal topic
discovery. This approach creates more semantically coherent topics
compared to traditional methods by leveraging deep learning embeddings.

## Usage

``` r
fit_embedding_topics(
  texts,
  method = "umap_hdbscan",
  n_topics = 10,
  embedding_model = "all-MiniLM-L6-v2",
  clustering_method = "kmeans",
  similarity_threshold = 0.7,
  min_topic_size = 3,
  umap_neighbors = 15,
  umap_min_dist = 0,
  umap_n_components = 5,
  representation_method = "c-tfidf",
  diversity = 0.5,
  reduce_outliers = TRUE,
  seed = 123,
  verbose = TRUE
)
```

## Arguments

- texts:

  A character vector of texts to analyze.

- method:

  The topic modeling method: "umap_hdbscan" (uses BERTopic),
  "embedding_clustering", "hierarchical_semantic".

- n_topics:

  The number of topics to identify. For UMAP+HDBSCAN, use NULL or "auto"
  for automatic determination, or specify an integer.

- embedding_model:

  The embedding model to use (default: "all-MiniLM-L6-v2").

- clustering_method:

  The clustering method for embedding-based approach: "kmeans",
  "hierarchical", "dbscan", "hdbscan".

- similarity_threshold:

  The similarity threshold for topic assignment (default: 0.7).

- min_topic_size:

  The minimum number of documents per topic (default: 3).

- umap_neighbors:

  The number of neighbors for UMAP dimensionality reduction (default:
  15).

- umap_min_dist:

  The minimum distance for UMAP (default: 0.0). Use 0.0 for tight,
  well-separated clusters. Use 0.1+ for visualization purposes. Range:
  0.0-0.99.

- umap_n_components:

  The number of UMAP components (default: 5).

- representation_method:

  The method for topic representation: "c-tfidf", "tfidf", "mmr",
  "frequency" (default: "c-tfidf").

- diversity:

  Topic diversity parameter between 0 and 1 (default: 0.5).

- reduce_outliers:

  Logical, if TRUE, reduces outliers in HDBSCAN clustering (default:
  TRUE).

- seed:

  Random seed for reproducibility (default: 123).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list containing topic assignments, topic keywords, and quality
metrics.

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech
  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )
  texts <- united_tbl$united_texts

  # Embedding-based topic modeling (powered by BERTopic)
  result <- TextAnalysisR::fit_embedding_topics(
    texts = texts,
    method = "umap_hdbscan",
    n_topics = 8,
    min_topic_size = 3
  )

  print(result$topic_assignments)
  print(result$topic_keywords)
}
```
