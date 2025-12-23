# Topic Modeling

Topic modeling discovers hidden themes in text collections.

## Setup

``` r
library(TextAnalysisR)

mydata <- SpecialEduTech
united_tbl <- unite_cols(mydata, listed_vars = c("title", "keyword", "abstract"))
tokens <- prep_texts(united_tbl, text_field = "united_texts")
dfm_object <- quanteda::dfm(tokens)
```

## Find Optimal Topics

``` r
find_optimal_k(dfm_object, topic_range = 5:30)
```

------------------------------------------------------------------------

**STM (Structural Topic Model)**

STM discovers latent topics using probabilistic modeling while
incorporating document metadata as covariates. It models how topics vary
across documents based on metadata like time, author, or category.

**Best For:**

- Metadata analysis: Relate topics to document characteristics
- Covariate effects: Test how metadata affects topics
- Interpretability: Clear word-probability distributions

**Quality Metrics:**

- **Semantic Coherence:** Measures word co-occurrence within topics
  (Mimno et al., 2011)
- **Exclusivity:** Measures how unique words are to each topic
- **Held-out Likelihood:** Predictive performance on unseen documents

``` r
out <- quanteda::convert(dfm_object, to = "stm")

model <- stm::stm(
  documents = out$documents,
  vocab = out$vocab,
  K = 15,
  prevalence = ~ reference_type + s(year),
  data = out$meta
)

terms <- get_topic_terms(model, top_term_n = 10)
```

**Learn More:** [Structural Topic
Model](https://www.structuraltopicmodel.com/) \| [STM
Vignette](https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf)

------------------------------------------------------------------------

**Embedding-based Topics (BERTopic)**

Uses transformer embeddings to capture semantic meaning, then applies
UMAP dimensionality reduction and HDBSCAN clustering to discover topics.
Creates semantically coherent topics based on deep contextual
understanding.

**Best For:**

- Semantic coherence: Capture meaning beyond word co-occurrence
- Short texts: Tweets, reviews, survey responses
- Multilingual: Handles multiple languages

**Key Components:**

- **Sentence Transformers:** Generate contextual embeddings (e.g.,
  all-MiniLM-L6-v2)
- **UMAP:** Dimensionality reduction preserving local structure
- **HDBSCAN:** Density-based clustering for topic discovery
- **c-TF-IDF:** Class-based TF-IDF for topic keyword extraction

``` r
results <- fit_embedding_topics(
  texts = united_tbl$united_texts,
  n_topics = 15
)
```

**Learn More:** [BERTopic](https://maartengr.github.io/BERTopic/) \|
[Sentence-BERT](https://www.sbert.net/)

------------------------------------------------------------------------

**Hybrid Topic Modeling**

Combines the strengths of both STM and embedding-based approaches. Uses
transformer embeddings for semantic understanding while maintaining
STM’s ability to model covariate relationships and provide probabilistic
topic assignments.

**Best For:**

- Best of both worlds: Need semantic coherence AND covariate modeling
- Complex research: Testing hypotheses about how metadata affects
  semantically-defined topics
- Validation: Compare and validate findings across different
  methodological approaches

**Quality Metrics:**

- **Semantic Coherence:** How often top words co-occur in documents
  (higher is better)
- **Exclusivity:** How unique words are to each topic (higher is better)
- **Silhouette Score:** Cluster separation for embedding topics (-1 to
  1, higher is better)
- **Alignment Score:** Agreement between STM and embedding topic
  assignments
- **Adjusted Rand Index:** Clustering agreement corrected for chance

``` r
results <- fit_hybrid_model(
  texts = united_tbl$united_texts,
  metadata = united_tbl[, c("reference_type", "year")],
  n_topics_stm = 15
)
```

**Learn More:** [Structural Topic
Model](https://www.structuraltopicmodel.com/) \|
[BERTopic](https://maartengr.github.io/BERTopic/)

------------------------------------------------------------------------

## AI Topic Labels

``` r
labels <- generate_topic_labels(
  terms,
  provider = "ollama"  # or "openai"
)
```

------------------------------------------------------------------------

**Methods Comparison**

| Feature          | STM           | Embedding     | Hybrid |
|------------------|---------------|---------------|--------|
| Speed            | Fast          | Medium        | Slow   |
| Metadata Support | Yes           | No            | Yes    |
| Semantic Meaning | Word patterns | Deep semantic | Both   |
| Interpretability | High          | Medium        | High   |
| Short Texts      | Poor          | Good          | Good   |
| Multilingual     | No            | Yes           | Yes    |
| Requires Python  | No            | Yes           | Yes    |

**When to Use Each:**

- **STM:** Academic research with metadata, covariate effects,
  interpretability priority
- **Embedding:** Short texts, multilingual, semantic similarity over
  frequency
- **Hybrid:** High-stakes research, methodological validation,
  comprehensive analysis

------------------------------------------------------------------------

## Next Steps

- [Semantic
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/semantic_analysis.md)
- [Python
  Environment](https://mshin77.github.io/TextAnalysisR/articles/python_environment.md)
  (for embedding-based methods)
