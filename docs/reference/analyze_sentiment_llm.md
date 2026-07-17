# LLM-based Sentiment Analysis

Analyzes sentiment using Large Language Models (OpenAI or Gemini).
Provides nuanced sentiment understanding including sarcasm detection,
mixed emotions, and contextual interpretation that lexicon-based methods
miss.

Requests use temperature 0, which makes outputs near-deterministic but
not fully reproducible across runs or model versions; record the
returned `model_used` field when reporting results.

## Usage

``` r
analyze_sentiment_llm(
  texts,
  doc_names = NULL,
  provider = c("openai", "gemini"),
  model = NULL,
  api_key = NULL,
  batch_size = 5,
  include_explanation = FALSE,
  verbose = TRUE
)
```

## Arguments

- texts:

  Character vector of texts to analyze.

- doc_names:

  Optional character vector of document names (default: text1, text2,
  ...).

- provider:

  AI provider to use: "openai" (default) or "gemini".

- model:

  Model name. If NULL, uses provider defaults: "gpt-4.1-mini" (OpenAI),
  "gemini-2.5-flash-lite" (Gemini).

- api_key:

  API key for OpenAI or Gemini. If NULL, uses environment variable.

- batch_size:

  Number of texts to process per API call (default: 5). Larger batches
  are more efficient but may hit token limits.

- include_explanation:

  Logical, if TRUE includes natural language explanation for each
  sentiment classification (default: FALSE).

- verbose:

  Logical, if TRUE prints progress messages (default: TRUE).

## Value

A list containing:

- document_sentiment:

  Data frame with document-level sentiment scores

- summary_stats:

  Summary statistics of the analysis

- model_used:

  Model name used for analysis

- provider:

  AI provider used

## Details

LLM-based sentiment analysis offers several advantages over lexicon
methods:

- Understands context and nuance

- Detects sarcasm and irony

- Handles mixed emotions

- Works across domains without retraining

## Examples

``` r
if (interactive()) {
  abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:5]

  sentiment_openai <- analyze_sentiment_llm(abstracts, provider = "openai")

  sentiment_gemini <- analyze_sentiment_llm(abstracts, provider = "gemini",
                                             include_explanation = TRUE)
}
```
