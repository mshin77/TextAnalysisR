# Generate Embeddings

Generates embeddings for texts using sentence transformers.

## Usage

``` r
generate_embeddings(texts, model = "all-MiniLM-L6-v2", verbose = TRUE)
```

## Arguments

- texts:

  A character vector of texts.

- model:

  Sentence transformer model name (default: "all-MiniLM-L6-v2").

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A matrix of embeddings.
