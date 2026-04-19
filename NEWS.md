# TextAnalysisR 0.1.3 (2026-04-18)

- Updated semantic analysis and topic modeling functions
- Refreshed vignettes and pkgdown reference index

# TextAnalysisR 0.0.3 (2026-02-22)

- Fixed citation and documentation year from 2025 to 2026
- Updated package title to "A Text Mining Workflow Tool" across documentation
- Added pkgdown site URL to DESCRIPTION
- Updated `plot_topic_probability()` documentation
- Refreshed pkgdown site articles and reference pages

# TextAnalysisR 0.0.3 (2025-12-27)

## Changes
- `fit_embedding_topics()` renamed to `fit_embedding_model()` for consistency with
  other topic modeling functions (`fit_hybrid_model()`, `fit_temporal_model()`).
  The old name still works but is deprecated.

## New Features
- Multi-format file import (PDF, DOCX, XLSX, CSV, TXT)
- Hybrid topic modeling (STM + BERTopic)
- Semantic similarity and document clustering
- Lexical diversity metrics

## Embedding Enhancements
- `get_best_embeddings()`: Auto-detects and uses best available provider
  - Priority: Ollama (local) > sentence-transformers (Python) > OpenAI/Gemini (API)
- Shared embeddings cache between Document Clustering and Topic Modeling
- Precomputed embeddings support for `fit_hybrid_model()` and `fit_embedding_model()`

## UI/UX Improvements
- Removed GPU checkbox from Sentiment Analysis for consistency
- Streamlined provider selection for AI features
- Improved error messages for missing prerequisites

## Lexical Analysis Enhancements
- Log odds ratio analysis for categorical word frequency comparisons
- Lexical dispersion plots showing term distribution across corpus
- Dispersion metrics quantifying how evenly terms spread across documents

## Semantic Analysis Enhancements
- Neural sentiment analysis via transformers (DistilBERT, RoBERTa)
- Network node attribute controls (size by degree/betweenness/frequency)
- Node color options (community detection/frequency gradient)

## NLP Integration
- spaCy: POS tagging, NER, lemmatization, dependency parsing
- Morphological analysis 
- Sentence transformers for document embeddings

## AI Integration
- Multi-provider: OpenAI, Gemini, Ollama (local)
- Topic-grounded content generation
- RAG search over document corpus
- Responsible AI design 

## Accessibility
- WCAG 2.1 Level AA compliant
- Dark mode, keyboard navigation, screen reader support
- Multi-language support 

## Security
- Rate limiting, input validation, API key protection

## Bug Fixes
- Removed internal functions from package exports
  (`run_neural_topics_internal`, `run_temporal_topics_internal`,
  `run_contrastive_topics_internal`, `calculate_eval_metrics_internal`)

# TextAnalysisR 0.0.2 (2024-12-05)

- Improved documentation
- Enhanced DESCRIPTION metadata
- CRAN policy compliance updates

# TextAnalysisR 0.0.1 (2023-10-18)

- Initial CRAN release
- Core text mining functionality
- STM topic modeling
- Text preprocessing capabilities

