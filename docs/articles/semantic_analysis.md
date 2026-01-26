# Semantic Analysis

Semantic analysis finds patterns of meaning using embeddings and neural
networks.

## Setup

``` r
library(TextAnalysisR)

mydata <- SpecialEduTech
united_tbl <- unite_cols(mydata, listed_vars = c("title", "keyword", "abstract"))
tokens <- prep_texts(united_tbl, text_field = "united_texts")
dfm_object <- quanteda::dfm(tokens)
```

## Document Similarity

``` r
similarity <- semantic_similarity_analysis(
  texts = united_tbl$united_texts,
  method = "cosine"
)
```

------------------------------------------------------------------------

**Similarity Methods**

Semantic analysis measures document similarity using different
approaches to capture meaning, from simple vocabulary matching to deep
neural representations.

**Methods:**

| Method | Description | Best For |
|----|----|----|
| Words | Lexical analysis using word frequency vectors (bag-of-words) | Finding documents with shared terminology |
| N-grams | Phrase-based analysis capturing word sequences | Detecting similar phraseology |
| Embeddings | Deep semantic analysis using transformer models | Conceptual similarity, handles synonyms |

**Usage:** Choose method based on your analysis goals. Words and n-grams
are faster and interpretable. Embeddings capture deeper meaning but
require more computation. All methods use cosine similarity for
comparison.

**Learn More:** [Sentence Transformers
Documentation](https://www.sbert.net/)

------------------------------------------------------------------------

## Sentiment Analysis

### Lexicon-based (no Python)

``` r
sentiment <- sentiment_lexicon_analysis(dfm_object, lexicon = "afinn")
plot_sentiment_distribution(sentiment$document_sentiment)
```

### Neural (requires Python)

``` r
sentiment <- sentiment_embedding_analysis(united_tbl$united_texts)
```

------------------------------------------------------------------------

## Document Clustering

> **Shiny App:** Document clustering is now in **Topic Modeling →
> Embedding-based Topics**, which combines clustering with automatic
> keyword extraction.

``` r
results <- fit_embedding_model(
  texts = united_tbl$united_texts,
  method = "umap_dbscan",
  backend = "r",
  n_topics = 5
)

results$topic_assignments
results$topic_keywords
```

For standalone clustering without keywords, use
[`cluster_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/cluster_embeddings.md).
See [Topic
Modeling](https://mshin77.github.io/TextAnalysisR/articles/topic_modeling.md)
for details.

### AI Cluster Labels

``` r
labels <- generate_cluster_labels(
  results$topic_keywords,
  provider = "ollama"
)
```

------------------------------------------------------------------------

**Algorithms Reference**

**Clustering:** K-means (spherical), Hierarchical (nested), DBSCAN
(density-based), HDBSCAN (auto-detect K)

**Dimensionality Reduction:** PCA (fast, linear), t-SNE (local
structure), UMAP (balanced)

------------------------------------------------------------------------

## Network Analysis

Visualize word relationships as interactive networks with community
detection.

### Word Co-occurrence Network

``` r
network <- word_co_occurrence_network(
  dfm_object,
  co_occur_n = 10,                    # Minimum co-occurrence count
  top_node_n = 30,                    # Top nodes to display
  node_label_size = 22,               # Font size (12-40)
  community_method = "leiden"         # Community detection algorithm
)

network$plot   # Interactive visNetwork plot
network$table  # Node metrics (degree, eigenvector, community)
network$stats  # 9 network statistics
```

### Word Correlation Network

``` r
corr_network <- word_correlation_network(
  dfm_object,
  common_term_n = 20,                 # Minimum term frequency
  corr_n = 0.4,                       # Minimum correlation threshold
  community_method = "leiden"
)
```

### Category-Specific Analysis

Enable per-category networks in the Shiny app to generate separate
networks for each category, displayed in a tabbed interface.

------------------------------------------------------------------------

**Network Statistics (9 Metrics)**

Each network returns comprehensive statistics:

| Metric | Description |
|----|----|
| Nodes | Total unique words in network |
| Edges | Total connections between nodes |
| Density | Proportion of possible edges present (0-1) |
| Diameter | Longest shortest path in network |
| Global Clustering | Overall network clustering tendency |
| Avg Local Clustering | Average of local clustering coefficients |
| Modularity | Quality of community structure (higher = better separation) |
| Assortativity | Tendency of similar nodes to connect |
| Avg Path Length | Average distance between nodes |

------------------------------------------------------------------------

**Community Detection Methods**

Community detection identifies clusters of semantically related nodes.

| Method | Description | Best For |
|----|----|----|
| `leiden` | Modern algorithm, guarantees well-connected communities | Default, best quality |
| `louvain` | Fast modularity optimization | Large networks |
| `label_prop` | Propagates labels through network | Very large networks |
| `fast_greedy` | Hierarchical agglomerative | Quick exploration |

**Learn More:** [igraph Community
Detection](https://igraph.org/r/doc/communities.html)

------------------------------------------------------------------------

## Temporal Analysis

Track themes over time:

``` r
temporal <- temporal_semantic_analysis(
  texts = united_tbl$united_texts,
  timestamps = united_tbl$year
)
```

------------------------------------------------------------------------

**Embedding Models**

| Model                   | Speed  | Quality | Use Case           |
|-------------------------|--------|---------|--------------------|
| all-MiniLM-L6-v2        | Fast   | Good    | General purpose    |
| all-mpnet-base-v2       | Slow   | Best    | Highest quality    |
| paraphrase-multilingual | Medium | Good    | Multiple languages |

**Learn More:** [Sentence Transformers
Models](https://www.sbert.net/docs/pretrained_models.html)

------------------------------------------------------------------------

## Next Steps

- [Topic
  Modeling](https://mshin77.github.io/TextAnalysisR/articles/topic_modeling.md)
