# Call Ollama for Text Generation

Sends a prompt to Ollama and returns the generated text.

## Usage

``` r
call_ollama(
  prompt,
  model = "llama3.2",
  temperature = 0.3,
  max_tokens = 512,
  timeout = 60,
  verbose = FALSE
)
```

## Arguments

- prompt:

  Character string containing the prompt.

- model:

  Character string specifying the Ollama model (default: "llama3.2").

- temperature:

  Numeric value controlling randomness (default: 0.3).

- max_tokens:

  Maximum number of tokens to generate (default: 512).

- timeout:

  Timeout in seconds for the request (default: 60).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

Character string with the generated text, or NULL if failed.

## See also

Other ai:
[`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md),
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md),
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
response <- call_ollama(
  prompt = "Summarize these keywords: machine learning, neural networks, AI",
  model = "llama3.2"
)
print(response)
} # }
```
