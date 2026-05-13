# Changelog

## TextAnalysisR 0.1.4

- New `math_mode` argument in
  [`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md)
  keeps numbers, math operators, and symbols, and strips only
  sentence-end punctuation.
- First CRAN release.

## TextAnalysisR 0.1.3

- Updated semantic analysis and topic modeling functions.
- Refreshed vignettes and pkgdown reference index.

## TextAnalysisR 0.1.0

- Renamed
  [`fit_embedding_topics()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_topics.md)
  to
  [`fit_embedding_model()`](https://mshin77.github.io/TextAnalysisR/reference/fit_embedding_model.md).
  The old name is deprecated.
- Multi-format file import (PDF, DOCX, XLSX, CSV, TXT).
- Hybrid topic modeling (STM + BERTopic).
- Semantic similarity and document clustering.
- Lexical diversity metrics (TTR, MTLD).
- Log-odds ratio analysis and lexical dispersion plots.
- Neural sentiment analysis via transformers.
- spaCy support for POS tagging, NER, lemmatization, and dependency
  parsing.
- Sentence-transformer document embeddings.
- LLM integration (OpenAI, Gemini, Ollama) for topic labeling and RAG.

## TextAnalysisR 0.0.2

- Documentation improvements.

## TextAnalysisR 0.0.1

- First development release.
- Text preprocessing, STM topic modeling, basic Shiny app.
