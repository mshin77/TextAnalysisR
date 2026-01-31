# Get Best Available Embeddings

Auto-detects and uses the best available embedding provider with the
following priority:

1.  Ollama (free, local, fast) - if running

2.  sentence-transformers (local Python) - if Python environment is set
    up

3.  OpenAI API - if OPENAI_API_KEY is set

4.  Gemini API - if GEMINI_API_KEY is set

## Usage

``` r
get_best_embeddings(
  texts,
  provider = "auto",
  model = NULL,
  api_key = NULL,
  verbose = TRUE
)
```

## Arguments

- texts:

  Character vector of texts to embed

- provider:

  Character string: "auto" (default), "ollama", "sentence-transformers",
  "openai", or "gemini". Use "auto" for automatic detection.

- model:

  Character string specifying the embedding model. If NULL, uses default
  model for the selected provider.

- api_key:

  Optional API key for OpenAI or Gemini providers. If NULL, falls back
  to environment variables (OPENAI_API_KEY, GEMINI_API_KEY).

- verbose:

  Logical, whether to print progress messages (default: TRUE)

## Value

Matrix with embeddings (rows = texts, columns = dimensions)

## See also

Other ai:
[`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md),
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`call_openai_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_openai_chat.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`describe_image()`](https://mshin77.github.io/TextAnalysisR/reference/describe_image.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`get_api_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_api_embeddings.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md)

## Examples

``` r
if (FALSE) { # \dontrun{
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:5]

# Auto-detect best available provider
embeddings <- get_best_embeddings(texts)

# Force specific provider
embeddings <- get_best_embeddings(texts, provider = "ollama")

dim(embeddings)
} # }
```
