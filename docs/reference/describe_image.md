# Describe Image Using Vision LLM

Unified dispatcher for image description using vision LLMs. Routes to
the appropriate provider (Ollama, OpenAI, or Gemini).

## Usage

``` r
describe_image(
  image_base64,
  provider = "ollama",
  model = NULL,
  api_key = NULL,
  prompt =
    "Describe this image in detail, focusing on any charts, diagrams, tables, or textual content. Extract any visible text.",
  timeout = 120
)
```

## Arguments

- image_base64:

  Character string of base64-encoded PNG image

- provider:

  Character: "ollama", "openai", or "gemini"

- model:

  Character: Model name (uses provider default if NULL)

- api_key:

  Character: API key (required for openai/gemini)

- prompt:

  Character: Description prompt

- timeout:

  Numeric: Request timeout in seconds (default: 120)

## Value

Character string description, or NULL on failure

## See also

Other ai:
[`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md),
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`call_openai_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_openai_chat.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`get_api_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_api_embeddings.md),
[`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md)
