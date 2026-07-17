# Generate Embeddings

Generates L2-normalized embeddings for texts using sentence
transformers.

## Usage

``` r
generate_embeddings(texts, model = "all-MiniLM-L6-v2", verbose = TRUE)
```

## Arguments

- texts:

  A character vector of texts.

- model:

  Sentence transformer model name (default: "all-MiniLM-L6-v2"). Newer
  small models (e.g. "BAAI/bge-small-en-v1.5") often score higher on
  retrieval benchmarks but expect instruction prefixes this function
  does not add.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A matrix of L2-normalized embeddings.
