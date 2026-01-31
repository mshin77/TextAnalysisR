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

Other ai:
[`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`call_openai_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_openai_chat.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`describe_image()`](https://mshin77.github.io/TextAnalysisR/reference/describe_image.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`get_api_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_api_embeddings.md),
[`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md)

## Examples

``` r
if (FALSE) { # \dontrun{
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
} # }
```
