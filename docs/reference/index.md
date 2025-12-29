# Package index

## Getting Started

Launch the app and import data

- [`run_app()`](https://mshin77.github.io/TextAnalysisR/reference/run_app.md)
  : Launch the TextAnalysisR app
- [`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md)
  : Process Files

## Preprocessing

Text preparation and feature extraction

- [`unite_cols()`](https://mshin77.github.io/TextAnalysisR/reference/unite_cols.md)
  : Unite Text Columns
- [`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md)
  : Preprocess Text Data
- [`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md)
  : Detect Multi-Word Expressions
- [`get_available_dfm()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_dfm.md)
  : Get Available Document-Feature Matrix with Fallback
- [`get_available_tokens()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_tokens.md)
  : Get Available Tokens with Fallback

## Linguistic Analysis

POS tagging, NER, and morphology (requires Python)

- [`extract_pos_tags()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pos_tags.md)
  : Extract Part-of-Speech Tags from Tokens
- [`extract_named_entities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_named_entities.md)
  : Extract Named Entities from Tokens
- [`extract_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/extract_morphology.md)
  : Extract Morphological Features
- [`summarize_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/summarize_morphology.md)
  : Summarize Morphology Features
- [`plot_pos_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_pos_frequencies.md)
  : Plot Part-of-Speech Tag Frequencies
- [`plot_entity_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_entity_frequencies.md)
  : Plot Named Entity Frequencies
- [`plot_morphology_feature()`](https://mshin77.github.io/TextAnalysisR/reference/plot_morphology_feature.md)
  : Plot Morphology Feature Distribution
- [`render_displacy_ent()`](https://mshin77.github.io/TextAnalysisR/reference/render_displacy_ent.md)
  : Render displaCy Entity Visualization

## Lexical Analysis

Word frequency, keywords, readability, and dispersion

- [`calculate_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_word_frequency.md)
  : Analyze and Visualize Word Frequencies Across a Continuous Variable
- [`lexical_analysis`](https://mshin77.github.io/TextAnalysisR/reference/lexical_analysis.md)
  : Lexical Analysis Functions
- [`extract_keywords_keyness()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_keyness.md)
  : Extract Keywords Using Statistical Keyness
- [`extract_keywords_tfidf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_tfidf.md)
  : Extract Keywords Using TF-IDF
- [`lexical_frequency_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_frequency_analysis.md)
  : Lexical Frequency Analysis
- [`lexical_diversity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_diversity_analysis.md)
  : Lexical Diversity Analysis
- [`clear_lexdiv_cache()`](https://mshin77.github.io/TextAnalysisR/reference/clear_lexdiv_cache.md)
  : Clear Lexical Diversity Cache
- [`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md)
  : Calculate Text Readability
- [`calculate_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_log_odds_ratio.md)
  : Calculate Log Odds Ratio Between Categories
- [`calculate_weighted_log_odds()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_weighted_log_odds.md)
  : Calculate Weighted Log Odds Ratio
- [`calculate_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_lexical_dispersion.md)
  : Calculate Lexical Dispersion
- [`calculate_dispersion_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_dispersion_metrics.md)
  : Calculate Dispersion Metrics
- [`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md)
  : Plot Word Frequency
- [`plot_tfidf_keywords()`](https://mshin77.github.io/TextAnalysisR/reference/plot_tfidf_keywords.md)
  : Plot TF-IDF Keywords
- [`plot_keyness_keywords()`](https://mshin77.github.io/TextAnalysisR/reference/plot_keyness_keywords.md)
  : Plot Statistical Keyness
- [`plot_keyword_comparison()`](https://mshin77.github.io/TextAnalysisR/reference/plot_keyword_comparison.md)
  : Plot Keyword Comparison (TF-IDF vs Frequency)
- [`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md)
  : Plot N-gram Frequency
- [`plot_mwe_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_mwe_frequency.md)
  : Plot Multi-Word Expression Frequency
- [`plot_readability_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_readability_distribution.md)
  : Plot Readability Distribution
- [`plot_readability_by_group()`](https://mshin77.github.io/TextAnalysisR/reference/plot_readability_by_group.md)
  : Plot Readability by Group
- [`plot_top_readability_documents()`](https://mshin77.github.io/TextAnalysisR/reference/plot_top_readability_documents.md)
  : Plot Top Documents by Readability
- [`plot_lexical_diversity_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_diversity_distribution.md)
  : Plot Lexical Diversity Distribution
- [`plot_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/plot_log_odds_ratio.md)
  : Plot Log Odds Ratio
- [`plot_weighted_log_odds()`](https://mshin77.github.io/TextAnalysisR/reference/plot_weighted_log_odds.md)
  : Plot Weighted Log Odds
- [`plot_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_dispersion.md)
  : Plot Lexical Dispersion

## Sentiment Analysis

Lexicon and embedding-based sentiment

- [`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md)
  : Analyze Text Sentiment
- [`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)
  : Analyze Sentiment Using Tidytext Lexicons
- [`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md)
  : Embedding-based Sentiment Analysis
- [`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md)
  : Plot Sentiment Box Plot by Category
- [`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md)
  : Plot Sentiment by Category
- [`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md)
  : Plot Sentiment Distribution
- [`plot_sentiment_violin()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_violin.md)
  : Plot Sentiment Violin Plot by Category
- [`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md)
  : Plot Emotion Radar Chart
- [`plot_document_sentiment_trajectory()`](https://mshin77.github.io/TextAnalysisR/reference/plot_document_sentiment_trajectory.md)
  : Plot Document Sentiment Trajectory
- [`get_sentiment_color()`](https://mshin77.github.io/TextAnalysisR/reference/get_sentiment_color.md)
  : Generate Sentiment Color Gradient
- [`get_sentiment_colors()`](https://mshin77.github.io/TextAnalysisR/reference/get_sentiment_colors.md)
  : Get Sentiment Color Palette

## Semantic Analysis

Similarity, clustering, and networks

- [`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md)
  : Generate Embeddings
- [`reduce_dimensions()`](https://mshin77.github.io/TextAnalysisR/reference/reduce_dimensions.md)
  : Dimensionality Reduction Analysis
- [`calculate_document_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_document_similarity.md)
  : Calculate Document Similarity
- [`calculate_similarity_robust()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_similarity_robust.md)
  : Calculate Similarity Robust
- [`calculate_cosine_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_cosine_similarity.md)
  : Calculate Cosine Similarity Matrix
- [`semantic_similarity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_similarity_analysis.md)
  : Semantic Similarity Analysis
- [`semantic_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/semantic_document_clustering.md)
  : Semantic Document Clustering
- [`cluster_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/cluster_embeddings.md)
  : Embedding-based Document Clustering
- [`analyze_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_document_clustering.md)
  : Analyze Document Clustering
- [`export_document_clustering()`](https://mshin77.github.io/TextAnalysisR/reference/export_document_clustering.md)
  : Export Document Clustering Analysis
- [`generate_cluster_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels.md)
  : Generate Cluster Label Suggestions (Human-in-the-Loop)
- [`generate_cluster_labels_auto()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels_auto.md)
  : Generate Cluster Labels
- [`temporal_semantic_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/temporal_semantic_analysis.md)
  : Temporal Semantic Analysis
- [`analyze_semantic_evolution()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_semantic_evolution.md)
  : Analyze Semantic Evolution
- [`word_co_occurrence_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_co_occurrence_network.md)
  : Analyze and Visualize Word Co-occurrence Networks
- [`word_correlation_network()`](https://mshin77.github.io/TextAnalysisR/reference/word_correlation_network.md)
  : Analyze and Visualize Word Correlation Networks
- [`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md)
  : Plot Semantic Analysis Visualization
- [`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md)
  : Plot Document Similarity Heatmap
- [`plot_cross_category_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cross_category_heatmap.md)
  : Plot Cross-Category Similarity Comparison
- [`plot_cluster_terms()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cluster_terms.md)
  : Plot Cluster Top Terms

## Topic Modeling

STM, embedding-based, and hybrid models

- [`find_optimal_k()`](https://mshin77.github.io/TextAnalysisR/reference/find_optimal_k.md)
  : Find Optimal Number of Topics
- [`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md)
  : Fit Embedding-based Topic Model
- [`fit_hybrid_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_hybrid_model.md)
  : Fit Hybrid Topic Model
- [`fit_semantic_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_semantic_model.md)
  : Fit Semantic Model
- [`fit_temporal_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_temporal_model.md)
  : Fit Temporal Topic Model
- [`auto_tune_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/auto_tune_embedding_topics.md)
  : Auto-tune BERTopic Hyperparameters
- [`assess_embedding_stability()`](https://mshin77.github.io/TextAnalysisR/reference/assess_embedding_stability.md)
  : Assess Embedding Topic Model Stability
- [`get_topic_terms()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_terms.md)
  : Select Top Terms for Each Topic
- [`get_topic_prevalence()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_prevalence.md)
  : Get Topic Prevalence (Gamma) from STM Model
- [`get_topic_texts()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_texts.md)
  : Convert Topic Terms to Text Strings
- [`calculate_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_probability.md)
  : Calculate Topic Probabilities
- [`calculate_topic_stability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_topic_stability.md)
  : Calculate Topic Stability
- [`identify_topic_trends()`](https://mshin77.github.io/TextAnalysisR/reference/identify_topic_trends.md)
  : Identify Topic Trends
- [`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md)
  : Generate Topic Labels Using OpenAI's API
- [`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md)
  : Generate Content from Topic Terms
- [`plot_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_probability.md)
  : Plot Per-Document Per-Topic Probabilities
- [`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md)
  : Plot Topic Effects for Categorical Variables
- [`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md)
  : Plot Topic Effects for Continuous Variables
- [`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md)
  : Plot Word Probabilities by Topic
- [`plot_model_comparison()`](https://mshin77.github.io/TextAnalysisR/reference/plot_model_comparison.md)
  : Plot Topic Model Comparison Scatter
- [`plot_quality_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/plot_quality_metrics.md)
  : Plot Topic Model Quality Metrics
- [`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md)
  : Plot Term Frequency Trends by Continuous Variable

## PDF & Multimodal

Text extraction from PDFs with optional vision AI

- [`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md)
  : Process PDF File (Unified Entry Point)
- [`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md)
  : Process PDF File
- [`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)
  : Process PDF File using Python
- [`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md)
  : Extract Text from PDF
- [`extract_text_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf_py.md)
  : Extract Text from PDF using Python
- [`extract_tables_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md)
  : Extract Tables from PDF using Python
- [`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md)
  : Extract PDF with Multimodal Analysis
- [`extract_pdf_smart()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_smart.md)
  : Smart PDF Extraction with Auto-Detection
- [`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md)
  : Detect PDF Content Type
- [`detect_pdf_content_type_py()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type_py.md)
  : Detect PDF Content Type using Python

## AI Integration

Topic-grounded content generation via local and web-based APIs

- [`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md)
  : Check if Ollama is Available
- [`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md)
  : List Available Ollama Models
- [`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md)
  : Call Ollama for Text Generation
- [`call_openai_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_openai_chat.md)
  : Call OpenAI Chat Completion API
- [`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md)
  : Call Gemini Chat API
- [`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md)
  : Call LLM API (Unified Wrapper)
- [`check_vision_models()`](https://mshin77.github.io/TextAnalysisR/reference/check_vision_models.md)
  : Check Vision Model Availability
- [`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md)
  : Get Recommended Ollama Model
- [`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md)
  : Get Best Available Embeddings
- [`get_api_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_api_embeddings.md)
  : Get Embeddings from API
- [`get_spacy_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_spacy_embeddings.md)
  : Get spaCy Word Embeddings
- [`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md)
  : RAG-Enhanced Semantic Search
- [`analyze_sentiment_llm()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment_llm.md)
  : LLM-based Sentiment Analysis
- [`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md)
  : Get Default System Prompt for Content Type
- [`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md)
  : Get Default User Prompt Template for Content Type

## Python Environment

Python environment setup

- [`setup_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/setup_python_env.md)
  : Setup Python Environment
- [`check_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/check_python_env.md)
  : Check Python Environment Status

## Validation

Quality metrics and cross-validation

- [`cross_analysis_validation()`](https://mshin77.github.io/TextAnalysisR/reference/cross_analysis_validation.md)
  : Cross Analysis Validation
- [`validate_cross_models()`](https://mshin77.github.io/TextAnalysisR/reference/validate_cross_models.md)
  : Cross-Analysis Validation
- [`validate_semantic_coherence()`](https://mshin77.github.io/TextAnalysisR/reference/validate_semantic_coherence.md)
  : Validate Semantic Coherence
- [`calculate_clustering_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_clustering_metrics.md)
  : Calculate Clustering Quality Metrics
- [`calculate_cross_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_cross_similarity.md)
  : Calculate Cross-Matrix Cosine Similarity
- [`analyze_similarity_gaps()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_similarity_gaps.md)
  : Analyze Similarity Gaps Between Categories
- [`extract_cross_category_similarities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_cross_category_similarities.md)
  : Extract Cross-Category Similarities from Full Similarity Matrix

## Data

Example datasets

- [`SpecialEduTech`](https://mshin77.github.io/TextAnalysisR/reference/SpecialEduTech.md)
  : Special education technology bibliographic data
- [`acronym`](https://mshin77.github.io/TextAnalysisR/reference/acronym.md)
  : Acronym List
- [`stm_15`](https://mshin77.github.io/TextAnalysisR/reference/stm_15.md)
  : An example structure of a structural topic model
- [`dictionary_list_1`](https://mshin77.github.io/TextAnalysisR/reference/dictionary_list_1.md)
  : Dictionary List 1
- [`dictionary_list_2`](https://mshin77.github.io/TextAnalysisR/reference/dictionary_list_2.md)
  : Dictionary List 2
- [`stopwords_list`](https://mshin77.github.io/TextAnalysisR/reference/stopwords_list.md)
  : Stopwords List
