# Generate Topic Labels Using AI

This function generates descriptive labels for each topic based on their
top terms using AI providers (OpenAI or Gemini).

## Usage

``` r
generate_topic_labels(
  top_topic_terms,
  provider = "auto",
  model = NULL,
  system = NULL,
  user = NULL,
  temperature = 0.5,
  api_key = NULL,
  openai_api_key = NULL,
  verbose = TRUE
)
```

## Arguments

- top_topic_terms:

  A data frame containing the top terms for each topic.

- provider:

  AI provider to use: "auto" (default), "openai", or "gemini". "auto"
  picks the first provider with an available API key.

- model:

  A character string specifying which model to use. If NULL, uses
  provider defaults: "gpt-4.1-mini" (OpenAI), "gemini-2.5-flash-lite"
  (Gemini).

- system:

  A character string containing the system prompt for the API. If NULL,
  the function uses the default system prompt.

- user:

  A character string containing the user prompt for the API. If NULL,
  the function uses the default user prompt.

- temperature:

  A numeric value controlling the randomness of the output (default:
  0.5).

- api_key:

  API key for OpenAI or Gemini. If NULL, uses environment variable.

- openai_api_key:

  Deprecated. Use `api_key` instead. Kept for backward compatibility.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A data frame containing the top terms for each topic along with their
generated labels.

## See also

[`get_topic_terms()`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_terms.md)
to extract top terms first;
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md)
for survey items / RQs / themes grounded in the same terms;
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md)
for the direct AI provider call

## Examples

``` r
if (interactive()) {
top_topic_terms <- get_topic_terms(stm_model, top_term_n = 10)

# Auto-detect provider (tries OpenAI -> Gemini)
labels <- generate_topic_labels(top_topic_terms)

# Use specific provider
labels_openai <- generate_topic_labels(top_topic_terms, provider = "openai")
labels_gemini <- generate_topic_labels(top_topic_terms, provider = "gemini")
}
```
