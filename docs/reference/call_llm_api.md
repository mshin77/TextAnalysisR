# Call LLM API (Unified Wrapper)

Unified wrapper for calling different LLM providers (OpenAI, Gemini,
Ollama). Automatically routes to the appropriate provider-specific
function.

## Usage

``` r
call_llm_api(
  provider = c("openai", "gemini", "ollama"),
  system_prompt,
  user_prompt,
  model = NULL,
  temperature = 0,
  max_tokens = 150,
  api_key = NULL
)
```

## Arguments

- provider:

  Character string: "openai", "gemini", or "ollama"

- system_prompt:

  Character string with system instructions

- user_prompt:

  Character string with user message

- model:

  Character string specifying the model (provider-specific defaults
  apply)

- temperature:

  Numeric temperature for response randomness (default: 0)

- max_tokens:

  Maximum number of tokens to generate (default: 150)

- api_key:

  Character string with API key (required for openai/gemini)

## Value

Character string with the model's response

## See also

[`sanitize_llm_input()`](https://mshin77.github.io/TextAnalysisR/reference/sanitize_llm_input.md)
to clean text before prompting;
[`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md)
for vector embeddings;
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md)
for RAG search (retrieval + generation)

## Examples

``` r
if (interactive()) {
# Using OpenAI
response <- call_llm_api(
  provider = "openai",
  system_prompt = "You are a helpful assistant.",
  user_prompt = "Generate a topic label",
  api_key = Sys.getenv("OPENAI_API_KEY")
)

# Using Gemini
response <- call_llm_api(
  provider = "gemini",
  system_prompt = "You are a helpful assistant.",
  user_prompt = "Generate a topic label",
  api_key = Sys.getenv("GEMINI_API_KEY")
)
}
```
