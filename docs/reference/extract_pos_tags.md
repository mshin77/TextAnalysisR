# Extract Part-of-Speech Tags from Tokens

Uses spaCy to extract part-of-speech (POS) tags from tokenized text.
Returns a data frame with token-level POS annotations.

## Usage

``` r
extract_pos_tags(
  tokens,
  include_lemma = TRUE,
  include_entity = FALSE,
  include_dependency = FALSE,
  model = "en_core_web_sm"
)
```

## Arguments

- tokens:

  A quanteda tokens object or character vector of texts.

- include_lemma:

  Logical; include lemmatized forms (default: TRUE).

- include_entity:

  Logical; include named entity recognition (default: FALSE).

- include_dependency:

  Logical; include dependency parsing (default: FALSE).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with columns:

- `doc_id`: Document identifier

- `sentence_id`: Sentence number within document

- `token_id`: Token position within sentence

- `token`: Original token

- `pos`: Universal POS tag (e.g., NOUN, VERB, ADJ)

- `tag`: Detailed POS tag (e.g., NN, VBD, JJ)

- `lemma`: Lemmatized form (if include_lemma = TRUE)

- `entity`: Named entity type (if include_entity = TRUE)

- `head_token_id`: Head token in dependency tree (if include_dependency
  = TRUE)

- `dep_rel`: Dependency relation type, e.g., nsubj, dobj (if
  include_dependency = TRUE)

## Details

This function requires the Python with spaCy installed. If spaCy is not
initialized, this function will attempt to initialize it with the
specified model.

## Examples

``` r
if (interactive()) {
  tokens <- quanteda::tokens(TextAnalysisR::SpecialEduTech$abstract[1])
  pos_data <- extract_pos_tags(tokens)
  print(pos_data)
}
```
