# Extract Named Entities with spaCy

Extract named entities from texts using spaCy NER.

## Usage

``` r
spacy_extract_entities(x, model = "en_core_web_sm")
```

## Arguments

- x:

  Character vector of texts OR a quanteda tokens object.

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with entity information:

- `doc_id`: Document identifier

- `text`: Entity text

- `label`: Entity type (PERSON, ORG, GPE, etc.)

- `start_char`: Start character position

- `end_char`: End character position
