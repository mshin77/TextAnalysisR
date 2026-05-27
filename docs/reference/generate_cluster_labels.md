# Generate Cluster Label Suggestions (Human-in-the-Loop)

Suggests descriptive labels for clusters using AI. Labels are
suggestions for human review - users should edit and approve before
using. Supports OpenAI, Gemini, or Ollama (local) for AI generation.
When running locally, Ollama is preferred for privacy and cost-free
operation.

## Usage

``` r
generate_cluster_labels(
  cluster_keywords,
  provider = "auto",
  model = NULL,
  temperature = 0.3,
  max_tokens = 50,
  api_key = NULL,
  verbose = TRUE
)
```

## Arguments

- cluster_keywords:

  List of keywords for each cluster.

- provider:

  AI provider to use: "auto" (default), "openai", "gemini", or "ollama".
  "auto" will try Ollama first, then check for OpenAI/Gemini keys.

- model:

  Model name. If NULL, uses provider defaults: "gpt-4.1-mini" (OpenAI),
  "gemini-2.5-flash-lite" (Gemini), or recommended Ollama model.

- temperature:

  Temperature parameter (default: 0.3).

- max_tokens:

  Maximum tokens for response (default: 50).

- api_key:

  API key for OpenAI or Gemini. If NULL, uses environment variable. Not
  required for Ollama.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list of generated labels.

## Examples

``` r
if (interactive()) {
  cluster_keywords <- list(
    "1" = c("calculator", "arithmetic", "elementary", "remedial"),
    "2" = c("computer", "instruction", "multiplication", "drill")
  )
  labels_ollama <- generate_cluster_labels(cluster_keywords, provider = "ollama")
  labels_openai <- generate_cluster_labels(cluster_keywords, provider = "openai")
  labels_gemini <- generate_cluster_labels(cluster_keywords, provider = "gemini")
}
```
