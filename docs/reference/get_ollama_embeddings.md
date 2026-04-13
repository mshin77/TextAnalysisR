# Get Ollama Embeddings (Internal)

Generate embeddings using local Ollama models.

## Usage

``` r
get_ollama_embeddings(texts, model = "nomic-embed-text")
```

## Arguments

- texts:

  Character vector of texts to embed.

- model:

  Ollama embedding model (default: "nomic-embed-text").

## Value

Numeric matrix of embeddings (one row per text).
