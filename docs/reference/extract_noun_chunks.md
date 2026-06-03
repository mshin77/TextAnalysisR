# Extract Noun Chunks

Extract noun chunks (base noun phrases) from texts. Useful for keyphrase
extraction.

## Usage

``` r
extract_noun_chunks(x, model = "en_core_web_sm")
```

## Arguments

- x:

  Character vector of texts OR a quanteda tokens object.

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with noun chunk information.
