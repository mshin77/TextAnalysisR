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

``` r
clusters <- semantic_document_clustering(
  texts = united_tbl$united_texts,
  n_clusters = 5
)

# AI-generated labels (optional)
labels <- generate_cluster_labels(
  clusters$cluster_keywords,
  provider = "ollama"  # or "openai"
)
```

------------------------------------------------------------------------

**Clustering Algorithms**

Clustering groups documents with similar semantic content into
categories. Documents within a cluster are more similar to each other
than to documents in other clusters.

**Algorithms:**

| Algorithm | Description | Use Case |
|----|----|----|
| K-means | Creates K spherical clusters | Fast, simple, requires specifying K |
| Hierarchical | Builds tree of clusters | Exploring nested structures |
| DBSCAN | Density-based, finds outliers | Arbitrarily shaped clusters |
| HDBSCAN | Hierarchical density-based | Auto-determines cluster count |

**Usage:** Choose discovery mode (Automatic, Manual, Advanced). Select
semantic feature space and algorithm. Automatic mode finds optimal
cluster count. Use visualizations and quality metrics to evaluate
results.

**Learn More:** [scikit-learn Clustering
Guide](https://scikit-learn.org/stable/modules/clustering.html)

------------------------------------------------------------------------

**Dimensionality Reduction**

Dimensionality reduction transforms high-dimensional data into 2D or 3D
visualizations while preserving the structure and relationships between
documents.

**Algorithms:**

| Algorithm | Description | Trade-offs |
|----|----|----|
| PCA | Principal Component Analysis, finds linear patterns | Fast, interpretable |
| t-SNE | Preserves local structure, reveals clusters | Slow, good for visualization |
| UMAP | Balances local and global structure | Faster than t-SNE, better topology |

**Usage:** Select a semantic feature space (words, n-grams, or
embeddings), then choose a reduction method. Adjust parameters
(perplexity, neighbors, dimensions) based on your data size and
structure. Use for visual exploration before clustering.

``` r
reduced <- reduce_dimensions(embeddings, method = "umap", n_components = 2)
plot_semantic_viz(reduced)
```

**Learn More:** [scikit-learn Manifold
Learning](https://scikit-learn.org/stable/modules/manifold.html)

------------------------------------------------------------------------

## Network Analysis

Visualize semantic relationships as interactive networks with community
detection.

### Feature Spaces

Networks support three feature spaces based on distributional semantics:

| Feature Space | Network Type | Nodes | Edges |
|----|----|----|----|
| **Words** | Word co-occurrence/correlation | Words | Frequency or correlation |
| **N-grams** | Phrase relationships | N-gram phrases | Frequency or correlation |
| **Embeddings** | Document similarity | Documents | Cosine similarity \> threshold |

### Word Co-occurrence Network

``` r
network <- semantic_cooccurrence_network(
  dfm_object,
  co_occur_n = 10,                    # Minimum co-occurrence count
  top_node_n = 30,                    # Top nodes to display
  node_label_size = 22,               # Font size (12-40)
  feature_type = "words",             # "words", "ngrams", or "embeddings"
  embedding_sim_threshold = 0.5,      # For embeddings: similarity cutoff
  community_method = "leiden"         # Community detection algorithm
)

network$plot   # Interactive visNetwork plot
network$table  # Node metrics (degree, eigenvector, community)
network$stats  # 9 network statistics
```

### Word Correlation Network

``` r
corr_network <- semantic_correlation_network(
  dfm_object,
  common_term_n = 20,                 # Minimum term frequency
  corr_n = 0.4,                       # Minimum correlation threshold
  feature_type = "words",
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
| Nodes | Total unique terms/documents in network |
| Edges | Total connections between nodes |
| Density | Proportion of possible edges present (0-1) |
| Diameter | Longest shortest path in network |
| Global Clustering | Overall network clustering tendency |
| Avg Local Clustering | Average of local clustering coefficients |
| Modularity | Quality of community structure (higher = better separation) |
| Assortativity | Tendency of similar nodes to connect |
| Avg Path Length | Average distance between nodes |

------------------------------------------------------------------------

**Embedding-Based Document Networks**

When `feature_type = "embeddings"`, networks show document-to-document
relationships:

- **Nodes** = Documents (not words)
- **Edges** = Document pairs with cosine similarity above threshold
- **Edge weight** = Similarity score

Adjust `embedding_sim_threshold` (0.3-0.9) to control network density: -
Higher threshold → fewer, stronger connections - Lower threshold → more
connections, may be noisy

``` r
doc_network <- semantic_cooccurrence_network(
  dfm_object,
  feature_type = "embeddings",
  embeddings = my_embeddings,         # Pre-computed embedding matrix
  embedding_sim_threshold = 0.6,      # Only connect highly similar docs
  top_node_n = 50
)
```

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
