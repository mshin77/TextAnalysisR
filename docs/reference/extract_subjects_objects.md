# Extract Subjects and Objects

Extract subject-verb-object (SVO) triples from texts using dependency
parsing.

## Usage

``` r
extract_subjects_objects(x, model = "en_core_web_sm")
```

## Arguments

- x:

  Character vector of texts OR a quanteda tokens object.

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with SVO information.
