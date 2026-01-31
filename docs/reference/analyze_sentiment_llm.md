# LLM-based Sentiment Analysis

Analyzes sentiment using Large Language Models (OpenAI, Gemini, or
Ollama). Provides nuanced sentiment understanding including sarcasm
detection, mixed emotions, and contextual interpretation that
lexicon-based methods miss.

## Usage

``` r
analyze_sentiment_llm(
  texts,
  doc_names = NULL,
  provider = c("openai", "gemini", "ollama"),
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

  AI provider to use: "openai" (default), "gemini", or "ollama".

- model:

  Model name. If NULL, uses provider defaults: "gpt-4.1-mini" (OpenAI),
  "gemini-2.5-flash" (Gemini), "llama3.2" (Ollama).

- api_key:

  API key for OpenAI or Gemini. If NULL, uses environment variable. Not
  required for Ollama.

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

## See also

Other sentiment:
[`analyze_sentiment()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_sentiment.md),
[`plot_document_sentiment_trajectory()`](https://mshin77.github.io/TextAnalysisR/reference/plot_document_sentiment_trajectory.md),
[`plot_emotion_radar()`](https://mshin77.github.io/TextAnalysisR/reference/plot_emotion_radar.md),
[`plot_sentiment_boxplot()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_boxplot.md),
[`plot_sentiment_by_category()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_by_category.md),
[`plot_sentiment_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_distribution.md),
[`plot_sentiment_violin()`](https://mshin77.github.io/TextAnalysisR/reference/plot_sentiment_violin.md),
[`sentiment_embedding_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_embedding_analysis.md),
[`sentiment_lexicon_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/sentiment_lexicon_analysis.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Using OpenAI
result <- analyze_sentiment_llm(
  texts = c("This product is amazing!", "Worst experience ever."),
  provider = "openai"
)

# Using Gemini with explanations
result <- analyze_sentiment_llm(
  texts = my_texts,
  provider = "gemini",
  include_explanation = TRUE
)

# Using local Ollama (free, no API key)
result <- analyze_sentiment_llm(
  texts = my_texts,
  provider = "ollama",
  model = "llama3"
)
} # }
```
