# Get spaCy Word Embeddings

Get word vector embeddings for words or texts using spaCy. Requires a
spaCy model with word vectors.

## Usage

``` r
get_spacy_embeddings(texts, model = "en_core_web_md")
```

## Arguments

- texts:

  Character vector of words or texts.

- model:

  Character; spaCy model to use (default: "en_core_web_md").

## Value

A matrix of word embeddings (rows = texts, cols = dimensions).
