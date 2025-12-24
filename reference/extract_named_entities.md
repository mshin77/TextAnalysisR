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

This function requires the spacyr package and a working Python
environment with spaCy installed. If spaCy is not initialized, this
function will attempt to initialize it with the specified model.

## See also

Other preprocessing:
[`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md),
[`extract_pos_tags()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pos_tags.md),
[`get_available_dfm()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_dfm.md),
[`get_available_tokens()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_tokens.md),
[`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md),
[`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md),
[`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md),
[`unite_cols()`](https://mshin77.github.io/TextAnalysisR/reference/unite_cols.md)

## Examples

``` r
if (FALSE) { # \dontrun{
tokens <- quanteda::tokens("Apple Inc. was founded by Steve Jobs in California.")
entity_data <- extract_named_entities(tokens)
print(entity_data)
} # }
```
