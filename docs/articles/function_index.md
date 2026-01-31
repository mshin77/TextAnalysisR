# Function Reference Cheatsheet

Quick reference guide organized by workflow stage.

## Quick Start Examples

### Complete Workflow (5 steps)

``` r
library(TextAnalysisR)

# 1. Load data
data(SpecialEduTech)
texts <- SpecialEduTech$abstract

# 2. Preprocess
tokens <- prep_texts(texts, remove_punct = TRUE, remove_numbers = TRUE)
dfm <- quanteda::dfm(tokens)

# 3. Analyze keywords
keywords <- extract_keywords_tfidf(dfm, top_n = 20)
plot_tfidf_keywords(keywords)

# 4. Topic modeling
model <- fit_embedding_model(texts, n_topics = 5)
get_topic_terms(model, n_terms = 10)

# 5. Sentiment analysis
sentiment <- sentiment_lexicon_analysis(texts, lexicon = "bing")
plot_sentiment_distribution(sentiment)
```

### Generate Embeddings

``` r
# Auto-detect best available provider
embeddings <- get_best_embeddings(texts)

# Reduce dimensions for visualization
reduced <- reduce_dimensions(embeddings, method = "umap", n_components = 2)
plot_semantic_viz(reduced)
```

### Network Analysis

``` r
# Co-occurrence network
word_co_occurrence_network(dfm, top_node_n = 30, co_occur_n = 5)

# Correlation network
word_correlation_network(dfm, top_node_n = 30, corr_n = 0.3)
```

------------------------------------------------------------------------

## 1. Data Import & Preprocessing

| Function | Purpose |
|----|----|
| [`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md) | Import CSV, XLSX, PDF, DOCX, TXT files |
| [`unite_cols()`](https://mshin77.github.io/TextAnalysisR/reference/unite_cols.md) | Combine multiple text columns into one |
| [`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md) | Tokenize with full preprocessing options |
| [`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md) | Find collocations (n-grams) |
| [`get_available_dfm()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_dfm.md) | Get best available DFM with fallback |

## 2. Lexical Analysis

| Function | Purpose |
|----|----|
| [`calculate_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_word_frequency.md) | Count word frequencies |
| [`extract_keywords_tfidf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_tfidf.md) | TF-IDF keyword extraction |
| [`extract_keywords_keyness()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_keyness.md) | Keyness-based keywords |
| [`lexical_diversity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_diversity_analysis.md) | TTR, MATTR, MTLD metrics |
| [`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md) | Flesch, SMOG, ARI scores |

**Visualization Functions**

| Function | Purpose |
|----|----|
| [`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md) | Bar chart of word frequencies |
| [`plot_tfidf_keywords()`](https://mshin77.github.io/TextAnalysisR/reference/plot_tfidf_keywords.md) | TF-IDF keyword visualization |
| [`plot_keyness_keywords()`](https://mshin77.github.io/TextAnalysisR/reference/plot_keyness_keywords.md) | Keyness comparison plot |
| [`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md) | N-gram frequency plot |
| [`plot_readability_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_readability_distribution.md) | Readability score distribution |
| [`plot_lexical_diversity_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_diversity_distribution.md) | Diversity metrics plot |

## 3. Sentiment Analysis

| Function | Purpose |
|----|----|
| [`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md) | Quick sentiment scoring |
| [`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md) | Dictionary-based (no Python) |
| [`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md) | Neural sentiment (Python) |
| [`analyze_sentiment_llm()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment_llm.md) | LLM-based with explanations (Ollama/OpenAI/Gemini) |

**Visualization Functions**

| Function | Purpose |
|----|----|
| [`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md) | Sentiment score histogram |
| [`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md) | Sentiment by group |
| [`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md) | Box plot comparison |
| [`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md) | Emotion radar chart |

## 4. Semantic Analysis

| Function | Purpose |
|----|----|
| [`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md) | Auto-detect and use best embedding provider |
| [`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md) | Create document embeddings (local) |
| [`reduce_dimensions()`](https://mshin77.github.io/TextAnalysisR/reference/reduce_dimensions.md) | PCA, t-SNE, UMAP reduction |
| [`calculate_document_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_document_similarity.md) | Compute similarity matrix |
| [`semantic_similarity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_similarity_analysis.md) | Full similarity workflow |
| [`semantic_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_document_clustering.md) | Cluster similar documents |
| [`generate_cluster_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels.md) | AI-generated cluster names |

**Visualization Functions**

| Function | Purpose |
|----|----|
| [`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md) | 2D/3D semantic visualization |
| [`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md) | Similarity matrix heatmap |
| [`plot_cross_category_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cross_category_heatmap.md) | Cross-category similarity comparison |
| [`plot_cluster_terms()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cluster_terms.md) | Cluster term visualization |

## 5. Network Analysis

| Function | Purpose |
|----|----|
| [`word_co_occurrence_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_co_occurrence_network.md) | Word co-occurrence graph |
| [`word_correlation_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_correlation_network.md) | Word correlation graph |

**Network Parameters**

| Parameter | Default | Description |
|----|----|----|
| `node_label_size` | 22 | Font size for node labels (12-40) |
| `community_method` | “leiden” | Algorithm: “leiden”, “louvain” |
| `top_node_n` | 30 | Number of top nodes to display |
| `co_occur_n` | 10 | Minimum co-occurrence count (co-occurrence only) |
| `corr_n` | 0.4 | Minimum correlation threshold (correlation only) |

**Network Statistics (9 Metrics)**

| Metric               | Description                      |
|----------------------|----------------------------------|
| Nodes                | Total unique words               |
| Edges                | Total connections                |
| Density              | Edge density (0-1)               |
| Diameter             | Longest shortest path            |
| Global Clustering    | Network clustering tendency      |
| Avg Local Clustering | Average local clustering         |
| Modularity           | Community structure quality      |
| Assortativity        | Similar node connection tendency |
| Avg Path Length      | Average node distance            |

## 6. Topic Modeling

| Function | Purpose |
|----|----|
| [`find_optimal_k()`](https://mshin77.github.io/TextAnalysisR/reference/find_optimal_k.md) | Search for optimal topic count |
| [`fit_semantic_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_semantic_model.md) | STM (Structural Topic Model) |
| [`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md) | Embedding-based topics (Python or R backend) |
| [`fit_hybrid_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_hybrid_model.md) | STM + embeddings hybrid |
| [`get_topic_terms()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_terms.md) | Extract top words per topic |
| [`get_topic_prevalence()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_prevalence.md) | Calculate topic prevalence |
| [`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md) | AI-generated topic names |

**Visualization Functions**

| Function | Purpose |
|----|----|
| [`plot_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_probability.md) | Topic probability distribution |
| [`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md) | Topic effects by category |
| [`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md) | Topic effects over continuous var |
| [`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md) | Word probability per topic |
| [`plot_quality_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/plot_quality_metrics.md) | Model quality metrics |

## 7. PDF Processing

| Function | Purpose |
|----|----|
| [`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md) | Auto-fallback: multimodal (R + vision LLM) then text-only |
| [`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md) | Extract text (R) |
| [`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md) | R-native vision AI for images in PDFs (Ollama/OpenAI/Gemini) |
| [`describe_image()`](https://mshin77.github.io/TextAnalysisR/reference/describe_image.md) | Describe an image using vision LLM |
| [`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md) | Detect PDF content type |

## 8. AI Integration

TextAnalysisR uses a human-in-the-loop approach where AI provides
suggestions that you review, edit, and approve before use. Content
generation is **topic-grounded**: drafts are based on validated topic
terms and beta scores, not parametric AI knowledge.

Supports local ([Ollama](https://ollama.com)) and web-based
([OpenAI](https://platform.openai.com/),
[Gemini](https://ai.google.dev/)) providers.

| Function | Purpose |
|----|----|
| [`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md) | Unified LLM API (all providers) |
| [`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md) | Local Ollama API |
| [`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md) | Gemini API |
| [`describe_image()`](https://mshin77.github.io/TextAnalysisR/reference/describe_image.md) | Vision LLM image description (Ollama/OpenAI/Gemini) |
| [`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md) | AI-suggested topic labels |
| [`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md) | Topic-grounded content drafts |
| [`generate_cluster_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels.md) | AI-suggested cluster names |
| [`analyze_sentiment_llm()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment_llm.md) | LLM-based sentiment analysis |
| [`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md) | RAG search over documents |
| [`get_api_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_api_embeddings.md) | Web-based embeddings (OpenAI, Gemini) |
| [`get_spacy_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_spacy_embeddings.md) | Local spaCy word embeddings |

**Ollama Utilities**

| Function | Purpose |
|----|----|
| [`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md) | Verify Ollama availability |
| [`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md) | List installed models |
| [`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md) | Auto-select best model |

## 9. Linguistic Analysis

| Function | Purpose |
|----|----|
| [`extract_pos_tags()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pos_tags.md) | Identify word types (nouns, verbs, adjectives) |
| [`extract_named_entities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_named_entities.md) | Find people, places, organizations in text |
| [`extract_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/extract_morphology.md) | Analyze verb tenses, plural forms |

Requires Python. Run
[`setup_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/setup_python_env.md)
first.

## 10. Python Environment

| Function | Purpose |
|----|----|
| [`setup_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/setup_python_env.md) | Set up Python environment |
| [`check_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/check_python_env.md) | Check Python configuration |

## 11. Validation & Quality

**Validation Functions**

| Function | Purpose |
|----|----|
| [`cross_analysis_validation()`](https://mshin77.github.io/TextAnalysisR/reference/cross_analysis_validation.md) | Cross-validate analysis |
| [`validate_semantic_coherence()`](https://mshin77.github.io/TextAnalysisR/reference/validate_semantic_coherence.md) | Check semantic coherence |
| [`calculate_clustering_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_clustering_metrics.md) | Clustering quality metrics |

------------------------------------------------------------------------

## Launch App

The Shiny app provides an interactive interface for all functions:

``` r
run_app()
```
