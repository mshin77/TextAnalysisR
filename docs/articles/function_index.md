# Function Reference Cheatsheet

Quick reference guide organized by workflow stage.

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
| [`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md) | Create document embeddings |
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
| [`plot_cluster_terms()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cluster_terms.md) | Cluster term visualization |

## 5. Network Analysis

| Function | Purpose |
|----|----|
| [`semantic_cooccurrence_network()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_cooccurrence_network.md) | Word/document co-occurrence graph |
| [`semantic_correlation_network()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_correlation_network.md) | Word/document correlation graph |

**Network Parameters**

| Parameter | Default | Description |
|----|----|----|
| `feature_type` | “words” | Feature space: “words”, “ngrams”, “embeddings” |
| `embedding_sim_threshold` | 0.5 | Similarity threshold for embedding networks (0.3-0.9) |
| `node_label_size` | 22 | Font size for node labels (12-40) |
| `community_method` | “leiden” | Algorithm: “leiden”, “louvain”, “label_prop”, “fast_greedy” |
| `top_node_n` | 30 | Number of top nodes to display |
| `co_occur_n` | 10 | Minimum co-occurrence count (co-occurrence only) |
| `corr_n` | 0.4 | Minimum correlation threshold (correlation only) |

**Network Statistics (9 Metrics)**

| Metric               | Description                      |
|----------------------|----------------------------------|
| Nodes                | Total unique terms/documents     |
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
| [`fit_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_topics.md) | Embedding-based topics (BERTopic) |
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
| [`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md) | Auto-fallback PDF extraction |
| [`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md) | Extract text (R) |
| [`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md) | Vision AI for images in PDFs |
| [`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md) | Detect PDF content type |

## 8. AI Integration

| Function | Purpose |
|----|----|
| [`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md) | Verify Ollama availability |
| [`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md) | Direct Ollama API call |
| [`call_openai_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_openai_chat.md) | OpenAI API call |
| [`generate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels_langgraph.md) | Multi-agent topic labeling |
| [`generate_survey_items()`](https://mshin77.github.io/TextAnalysisR/reference/generate_survey_items.md) | Generate survey items |

## 9. NLP with spaCy (Python)

| Function | Purpose |
|----|----|
| [`init_spacy_nlp()`](https://mshin77.github.io/TextAnalysisR/reference/init_spacy_nlp.md) | Initialize spaCy with model selection |
| [`spacy_parse_full()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_parse_full.md) | Full parsing with POS, NER, morphology |
| [`spacy_extract_entities()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_entities.md) | Extract named entities (span-level) |
| [`spacy_extract_noun_chunks()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_noun_chunks.md) | Extract noun phrases |
| [`spacy_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_similarity.md) | Semantic similarity between texts |
| [`spacy_model_info()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_model_info.md) | Get loaded model information |

**Morphological Features**

| Feature          | Description        | Example Values      |
|------------------|--------------------|---------------------|
| `morph_Number`   | Grammatical number | Sing, Plur          |
| `morph_Tense`    | Verb tense         | Past, Pres, Fut     |
| `morph_VerbForm` | Verb form          | Fin, Inf, Part, Ger |
| `morph_Person`   | Grammatical person | 1, 2, 3             |
| `morph_Case`     | Grammatical case   | Nom, Acc, Dat, Gen  |

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
