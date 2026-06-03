# Calculate Word Similarity

Calculate semantic similarity between two words using word vectors.
Requires a spaCy model with word vectors (en_core_web_md or
en_core_web_lg).

## Usage

``` r
get_word_similarity(word1, word2, model = "en_core_web_md")
```

## Arguments

- word1:

  Character; first word.

- word2:

  Character; second word.

- model:

  Character; spaCy model to use (default: "en_core_web_md").

## Value

A list with similarity score and metadata.
