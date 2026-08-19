# Changelog

## TextAnalysisR 0.1.4.9000 (development)

- Inter-coder agreement reports the unit count each statistic used.
- Coded assignments export as CSV or Excel.
- Added
  [`estimate_topic_effects()`](https://mshin77.github.io/TextAnalysisR/reference/estimate_topic_effects.md)
  for STM prevalence effects.
- Fixed neural-classifier sentiment scoring.
- Prevented the hosted app from crashing on large datasets.

## TextAnalysisR 0.1.4

CRAN release: 2026-07-27

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
- LLM integration (OpenAI, Gemini) for topic labeling and RAG.

## TextAnalysisR 0.0.2

- Documentation improvements.

## TextAnalysisR 0.0.1

- First development release.
- Text preprocessing, STM topic modeling, basic Shiny app.
