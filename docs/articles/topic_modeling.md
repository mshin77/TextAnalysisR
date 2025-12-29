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
results <- fit_embedding_model(
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

AI generates label suggestions based on topic terms. You review and
edit:

1.  **Generate**: AI creates draft labels from top terms
2.  **Review**: Examine suggestions in the output table
3.  **Edit**: Modify any labels that need refinement
4.  **Override**: Use manual labels field to replace AI suggestions

``` r
# AI suggests, human decides
labels <- generate_topic_labels(
  terms,
  provider = "ollama"  # or "openai"
)
# Review and edit labels before final use
```

## Topic-Grounded Content Generation

Generate draft content grounded in your topic model results rather than
AI parametric knowledge. The LLM receives topic labels and term
probabilities (beta scores), ensuring outputs are anchored to your data.

### Workflow

1.  **Run topic modeling** and optionally generate/edit topic labels
2.  **Select content type**: survey items, research questions, theme
    descriptions, policy recommendations, or interview questions
3.  **Generate drafts**: AI creates content using your top terms ordered
    by beta scores
4.  **Review and edit**: Examine all outputs before use
5.  **Export**: Download as CSV or Excel

### Content Types

| Type                  | Output                    | Use Case             |
|-----------------------|---------------------------|----------------------|
| Survey Item           | Likert-scale statement    | Scale development    |
| Research Question     | RQ for literature review  | Systematic reviews   |
| Theme Description     | Qualitative theme summary | Thematic analysis    |
| Policy Recommendation | Action-oriented statement | Policy analysis      |
| Interview Question    | Open-ended question       | Qualitative research |

### Example

``` r
# Get topic terms with beta scores
top_terms <- get_topic_terms(model, top_term_n = 10)

# Optional: Add topic labels
topic_labels <- c("1" = "Digital Learning Tools", "2" = "Family Engagement")

# Generate content grounded in topic terms
content <- generate_topic_content(
  topic_terms_df = top_terms,
  content_type = "survey_item",
  topic_labels = topic_labels,  # Optional
  provider = "ollama",
  model = "llama3"
)

# Review before use
print(content)
```

------------------------------------------------------------------------

**Prompt Format**

The LLM receives structured prompts with your topic data:

    Topic: Digital Learning Tools

    Top Terms (highest to lowest beta score):
    virtual (.035)
    manipulatives (.022)
    mathematical (.014)
    solving (.013)
    learning (.012)

This “topic-grounded” approach ensures content reflects your actual
topic model results, not generic AI knowledge. Prompts include
guidelines for person-first language and content-specific best
practices.

**Customizing Prompts:**

``` r
# Get default prompts
system_prompt <- get_content_type_prompt("survey_item")
user_template <- get_content_type_user_template("survey_item")

# Use custom prompts
content <- generate_topic_content(
  topic_terms_df = top_terms,
  content_type = "custom",
  system_prompt = "You are a survey methodology expert...",
  user_prompt_template = "Create an item for: {topic_label}\nTerms: {terms}"
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

## References

**Structural Topic Model (STM):**

- Roberts, M. E., Stewart, B. M., & Tingley, D. (2019). stm: An R
  package for structural topic models. *Journal of Statistical
  Software*, *91*(2), 1–40. <https://doi.org/10.18637/jss.v091.i02>

**Embedding-based Topic Modeling:**

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a
  class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
  <https://arxiv.org/abs/2203.05794>
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings
  using Siamese BERT-networks. In *Proceedings of the 2019 Conference on
  Empirical Methods in Natural Language Processing* (pp. 3982–3992).
  Association for Computational Linguistics.
  <https://arxiv.org/abs/1908.10084>
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold
  approximation and projection for dimension reduction. *arXiv preprint
  arXiv:1802.03426*. <https://arxiv.org/abs/1802.03426>
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based
  clustering based on hierarchical density estimates. In *Advances in
  Knowledge Discovery and Data Mining. PAKDD 2013. Lecture Notes in
  Computer Science* (Vol. 7819, pp. 160–172). Springer.
  <https://doi.org/10.1007/978-3-642-37456-2_14>

**Quality Metrics:**

- Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A.
  (2011). Optimizing semantic coherence in topic models. In *Proceedings
  of the 2011 Conference on Empirical Methods in Natural Language
  Processing* (pp. 262–272). Association for Computational Linguistics.
  <https://aclanthology.org/D11-1024/>
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the
  interpretation and validation of cluster analysis. *Journal of
  Computational and Applied Mathematics*, *20*, 53–65.
  <https://doi.org/10.1016/0377-0427(87)90125-7>
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of
  Classification*, *2*(1), 193–218. <https://doi.org/10.1007/BF01908075>

------------------------------------------------------------------------

## Next Steps

- [Semantic
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/semantic_analysis.md)
- [Python
  Environment](https://mshin77.github.io/TextAnalysisR/articles/python_environment.md)
  (for embedding-based methods)
