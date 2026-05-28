# Get Embeddings from API

Generates text embeddings using OpenAI or Gemini embedding APIs.

## Usage

``` r
get_api_embeddings(
  texts,
  provider = c("openai", "gemini"),
  model = NULL,
  api_key = NULL,
  batch_size = 100
)
```

## Arguments

- texts:

  Character vector of texts to embed

- provider:

  Character string: "openai" or "gemini"

- model:

  Character string specifying the embedding model. Defaults:

  - openai: "text-embedding-3-small"

  - gemini: "gemini-embedding-001"

- api_key:

  Character string with API key

- batch_size:

  Integer, number of texts to embed per API call (default: 100)

## Value

Matrix with embeddings (rows = texts, columns = dimensions)
