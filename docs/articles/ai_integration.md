# AI Integration

TextAnalysisR provides comprehensive AI/NLP capabilities via local and
web-based providers.

## Supported Providers

| Provider | Type | API Key | Best For |
|----|----|----|----|
| [Ollama](https://ollama.com) | Local | None | Privacy, no cost, offline use |
| [OpenAI](https://platform.openai.com/) | Web-based | OPENAI_API_KEY | Quality, speed |
| [Gemini](https://ai.google.dev/) | Web-based | GEMINI_API_KEY | Quality, speed |
| [spaCy](https://spacy.io/) | Local | None | Linguistic analysis |
| Transformers | Local | None | Embeddings, sentiment |

## Feature Categories

### 1. Topic-Grounded Content Generation

Generate content grounded in your validated topic terms (not generic AI
knowledge):

| Function | Purpose |
|----|----|
| [`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md) | AI-suggested labels from topic model terms |
| [`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md) | Survey items, research questions, theme descriptions, policy recommendations |

Content types available:

- **survey_item**: Likert-scale questionnaire items
- **research_question**: Literature review questions
- **theme_description**: Academic theme summaries
- **policy_recommendation**: Actionable policy suggestions
- **interview_question**: Open-ended interview prompts

``` r
# Generate topic labels
labels <- generate_topic_labels(
 top_topic_terms,
 provider = "ollama",
 model = "phi3:mini"
)

# Generate survey items
items <- generate_topic_content(
 topic_terms_df,
 content_type = "survey_item",
 provider = "openai"
)
```

### 2. Semantic Analysis & Clustering

| Function | Purpose |
|----|----|
| [`generate_cluster_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_cluster_labels.md) | AI-suggested names for document clusters |
| [`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md) | Question-answering over your corpus (RAG) |
| [`get_api_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_api_embeddings.md) | Web-based document embeddings (OpenAI, Gemini) |
| [`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md) | Local embeddings (sentence-transformers) |

``` r
# Generate cluster labels
cluster_labels <- generate_cluster_labels(
 cluster_keywords,
 provider = "auto"  # Tries Ollama first, then web-based APIs
)

# RAG search over your documents
result <- run_rag_search(
 query = "What are the main findings?",
 documents = my_docs,
 provider = "openai"
)
```

### 3. Sentiment Analysis

| Function | Type | Features |
|----|----|----|
| [`analyze_sentiment_llm()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment_llm.md) | LLM-based | Context-aware, detects sarcasm, mixed emotions |
| [`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md) | Local transformers | No API required, fast batch processing |
| [`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md) | Dictionary-based | Multiple lexicons (AFINN, Bing, NRC) |

``` r
# LLM-based sentiment (nuanced)
sentiment <- analyze_sentiment_llm(
 texts,
 provider = "gemini",
 include_explanation = TRUE
)

# Local transformer sentiment (no API needed)
sentiment <- sentiment_embedding_analysis(texts)

# Lexicon-based sentiment
sentiment <- sentiment_lexicon_analysis(texts, lexicon = "nrc")
```

### 4. Linguistic Analysis (spaCy)

Deep linguistic processing via spaCy NLP models:

| Function | Purpose |
|----|----|
| [`spacy_parse_full()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_parse_full.md) | Full annotation: POS, lemma, NER, dependency, morphology |
| [`extract_noun_chunks()`](https://mshin77.github.io/TextAnalysisR/reference/extract_noun_chunks.md) | Keyphrase extraction |
| [`extract_subjects_objects()`](https://mshin77.github.io/TextAnalysisR/reference/extract_subjects_objects.md) | Subject-verb-object triples |
| [`get_word_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/get_word_similarity.md) | Word vector similarity |
| [`find_similar_words()`](https://mshin77.github.io/TextAnalysisR/reference/find_similar_words.md) | Find semantically similar words |
| [`get_sentences()`](https://mshin77.github.io/TextAnalysisR/reference/get_sentences.md) | Sentence segmentation |

``` r
# Initialize spaCy
init_spacy_nlp("en_core_web_sm")

# Full linguistic parsing
parsed <- spacy_parse_full(
 texts,
 pos = TRUE,
 lemma = TRUE,
 entity = TRUE,
 dependency = TRUE,
 morph = TRUE
)

# Extract noun chunks (keyphrases)
chunks <- extract_noun_chunks(texts)

# Extract subject-verb-object triples
svo <- extract_subjects_objects(texts)
```

### 5. LLM API Access

Unified interface for all providers:

``` r
# Provider-agnostic (recommended)
response <- call_llm_api(
 provider = "openai",
 system_prompt = "You are a helpful assistant.",
 user_prompt = "Summarize this text..."
)

# Provider-specific
call_openai_chat(system_prompt, user_prompt, model = "gpt-4o-mini")
call_gemini_chat(system_prompt, user_prompt, model = "gemini-2.0-flash")
call_ollama(prompt, model = "phi3:mini")
```

### 6. Ollama Utilities

| Function | Purpose |
|----|----|
| [`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md) | Verify Ollama server is running |
| [`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md) | List installed models |
| [`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md) | Auto-select optimal model |

``` r
# Check if Ollama is available
if (check_ollama()) {
 models <- list_ollama_models()
 best_model <- get_recommended_ollama_model()
}
```

## Responsible AI Design

All AI features follow [NIST AI Risk Management
Framework](https://www.nist.gov/itl/ai-risk-management-framework)
principles:

| Principle       | Implementation                                    |
|-----------------|---------------------------------------------------|
| Human oversight | AI suggests, you review and approve               |
| User control    | Edit, regenerate, or override any output          |
| Transparency    | View prompts and parameters used                  |
| Privacy         | Local options (Ollama, spaCy) for sensitive data  |
| Grounding       | Content based on your data, not generic knowledge |

## Setup

### Local AI (Ollama)

``` r
# 1. Install Ollama: https://ollama.com
# 2. Pull a model (in terminal):
#    ollama pull phi3:mini
#    ollama pull llama3.1:8b

# 3. Verify in R:
check_ollama()
list_ollama_models()
```

### Web-based AI (OpenAI/Gemini)

``` r
# Set API keys (choose one or both)
Sys.setenv(OPENAI_API_KEY = "your-openai-key")
Sys.setenv(GEMINI_API_KEY = "your-gemini-key")

# Or use .env file in project root
# OPENAI_API_KEY=your-key
# GEMINI_API_KEY=your-key
```

### Linguistic Analysis (spaCy)

``` r
# Install Python dependencies
setup_python_env()

# Initialize spaCy with a model
init_spacy_nlp("en_core_web_sm")  # Small model
init_spacy_nlp("en_core_web_md")
```

## Default Models

| Provider | Chat Model       | Embedding Model        |
|----------|------------------|------------------------|
| OpenAI   | gpt-4o-mini      | text-embedding-3-small |
| Gemini   | gemini-2.0-flash | text-embedding-004     |
| Ollama   | phi3:mini        | \-                     |
| Local    | \-               | all-MiniLM-L6-v2       |
