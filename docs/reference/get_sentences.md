# Get Sentences

Segment texts into sentences using spaCy's sentence boundary detection.

## Usage

``` r
get_sentences(x, model = "en_core_web_sm")
```

## Arguments

- x:

  Character vector of texts OR a quanteda tokens object.

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with sentence information.
