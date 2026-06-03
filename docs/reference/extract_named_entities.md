# Extract Named Entities from Tokens

Uses spaCy to extract named entities (NER) from tokenized text. Returns
a data frame with token-level entity annotations.

## Usage

``` r
extract_named_entities(
  tokens,
  include_pos = TRUE,
  include_lemma = TRUE,
  model = "en_core_web_sm"
)
```

## Arguments

- tokens:

  A quanteda tokens object or character vector of texts.

- include_pos:

  Logical; include POS tags (default: TRUE).

- include_lemma:

  Logical; include lemmatized forms (default: TRUE).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with columns:

- `doc_id`: Document identifier

- `token`: Original token

- `entity`: Named entity type (e.g., PERSON, ORG, GPE)

- `pos`: Universal POS tag (if include_pos = TRUE)

- `lemma`: Lemmatized form (if include_lemma = TRUE)

## Details

This function requires the Python with spaCy installed. If spaCy is not
initialized, this function will attempt to initialize it with the
specified model.

## Examples

``` r
if (interactive()) {
  tokens <- quanteda::tokens(TextAnalysisR::SpecialEduTech$abstract[1])
  entity_data <- extract_named_entities(tokens)
  print(entity_data)
}
```
