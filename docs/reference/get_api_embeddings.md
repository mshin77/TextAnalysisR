# Get Embeddings from API

Generates text embeddings using Ollama (local), OpenAI, or Gemini
embedding APIs.

## Usage

``` r
get_api_embeddings(
  texts,
  provider = c("ollama", "openai", "gemini"),
  model = NULL,
  api_key = NULL,
  batch_size = 100
)
```

## Arguments

- texts:

  Character vector of texts to embed

- provider:

  Character string: "ollama" (local API, free), "openai", or "gemini"

- model:

  Character string specifying the embedding model. Defaults:

  - ollama: "nomic-embed-text"

  - openai: "text-embedding-3-small"

  - gemini: "gemini-embedding-001"

- api_key:

  Character string with API key (not required for Ollama)

- batch_size:

  Integer, number of texts to embed per API call (default: 100)

## Value

Matrix with embeddings (rows = texts, columns = dimensions)

## See also

Other ai:
[`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md),
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`call_openai_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_openai_chat.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md),
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

# Using local Ollama API (free, no API key required)
embeddings <- get_api_embeddings(texts, provider = "ollama")

# Using OpenAI API
embeddings <- get_api_embeddings(texts, provider = "openai")

dim(embeddings)
} # }
```
