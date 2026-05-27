# Parse Texts with spaCy

Parse texts using spaCy and return token-level annotations. This is the
main parsing function for NLP analysis. Works with character vectors or
quanteda tokens objects.

## Usage

``` r
spacy_parse_full(
  x,
  pos = TRUE,
  tag = TRUE,
  lemma = TRUE,
  entity = FALSE,
  dependency = FALSE,
  morph = FALSE,
  model = "en_core_web_sm"
)
```

## Arguments

- x:

  Character vector of texts OR a quanteda tokens object.

- pos:

  Logical; include coarse POS tags (default: TRUE).

- tag:

  Logical; include fine-grained tags (default: TRUE).

- lemma:

  Logical; include lemmatized forms (default: TRUE).

- entity:

  Logical; include named entity tags (default: FALSE).

- dependency:

  Logical; include dependency relations (default: FALSE).

- morph:

  Logical; include morphological features (default: FALSE).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with token-level annotations including:

- `doc_id`: Document identifier

- `sentence_id`: Sentence number within document

- `token_id`: Token position within sentence

- `token`: Original token text

- `pos`: Coarse POS tag (if pos = TRUE)

- `tag`: Fine-grained tag (if tag = TRUE)

- `lemma`: Lemmatized form (if lemma = TRUE)

- `entity`: Named entity tag (if entity = TRUE)

- `head_token_id`: Head token ID (if dependency = TRUE)

- `dep_rel`: Dependency relation (if dependency = TRUE)

- `morph`: Morphological features string (if morph = TRUE)

## Examples

``` r
if (interactive()) {
# From SpecialEduTech dataset
texts <- TextAnalysisR::SpecialEduTech$abstract[1:5]
parsed <- spacy_parse_full(texts, morph = TRUE)

# From quanteda tokens
united <- unite_cols(TextAnalysisR::SpecialEduTech, c("title", "abstract"))
tokens <- prep_texts(united, text_field = "united_texts")
parsed <- spacy_parse_full(tokens, morph = TRUE)
}
```
