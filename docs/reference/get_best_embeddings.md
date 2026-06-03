# Get Best Available Embeddings

Auto-detects and uses the best available embedding provider with the
following priority:

1.  sentence-transformers (local Python) - if Python environment is set
    up

2.  OpenAI API - if OPENAI_API_KEY is set

3.  Gemini API - if GEMINI_API_KEY is set

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

  Character string: "auto" (default), "sentence-transformers", "openai",
  or "gemini". Use "auto" for automatic detection.

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

## Examples

``` r
if (interactive()) {
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:5]

# Auto-detect best available provider
embeddings <- get_best_embeddings(texts)

# Force specific provider
embeddings <- get_best_embeddings(texts, provider = "openai")

dim(embeddings)
}
```
