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
