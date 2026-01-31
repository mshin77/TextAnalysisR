# Call OpenAI Chat Completion API

Makes a chat completion request to OpenAI's API.

## Usage

``` r
call_openai_chat(
  system_prompt,
  user_prompt,
  model = "gpt-4.1-mini",
  temperature = 0,
  max_tokens = 150,
  api_key
)
```

## Arguments

- system_prompt:

  Character string with system instructions

- user_prompt:

  Character string with user message

- model:

  Character string specifying the model (default: "gpt-4.1-mini")

- temperature:

  Numeric temperature for response randomness (default: 0)

- max_tokens:

  Maximum number of tokens to generate (default: 150)

- api_key:

  Character string with OpenAI API key

## Value

Character string with the model's response

## See also

Other ai:
[`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md),
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
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
response <- call_openai_chat(
  system_prompt = "You are a helpful assistant.",
  user_prompt = "Generate a topic label for: education, student, learning",
  api_key = Sys.getenv("OPENAI_API_KEY")
)
} # }
```
