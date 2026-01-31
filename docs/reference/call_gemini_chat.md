# Call Gemini Chat API

Makes a chat completion request to Google's Gemini API.

## Usage

``` r
call_gemini_chat(
  system_prompt,
  user_prompt,
  model = "gemini-2.5-flash",
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

  Character string specifying the Gemini model (default:
  "gemini-2.5-flash")

- temperature:

  Numeric temperature for response randomness (default: 0)

- max_tokens:

  Maximum number of tokens to generate (default: 150)

- api_key:

  Character string with Gemini API key

## Value

Character string with the model's response

## See also

Other ai:
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md),
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
response <- call_gemini_chat(
  system_prompt = "You are a helpful assistant.",
  user_prompt = "Generate a topic label for: education, student, learning",
  api_key = Sys.getenv("GEMINI_API_KEY")
)
} # }
```
