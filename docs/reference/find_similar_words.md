# Find Similar Words

Find words most similar to a given word using word vectors. Requires a
spaCy model with word vectors (en_core_web_md or en_core_web_lg).

## Usage

``` r
find_similar_words(word, top_n = 10L, model = "en_core_web_md")
```

## Arguments

- word:

  Character; target word.

- top_n:

  Integer; number of similar words to return (default: 10).

- model:

  Character; spaCy model to use (default: "en_core_web_md").

## Value

A data frame with similar words and similarity scores.
