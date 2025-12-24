# Extract Named Entities

Extract named entities at the span level using spaCy.

## Usage

``` r
spacy_extract_entities(texts, model = "en_core_web_sm")
```

## Arguments

- texts:

  Character vector of texts.

- model:

  Character; spaCy model to use.

## Value

A data.frame with columns:

- `doc_id`: Document identifier

- `text`: Entity text

- `label`: Entity type (PERSON, ORG, GPE, DATE, etc.)

- `start_char`: Start character position

- `end_char`: End character position

## See also

Other nlp:
[`spacy_extract_noun_chunks()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_noun_chunks.md),
[`spacy_model_info()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_model_info.md),
[`spacy_nlp`](https://mshin77.github.io/TextAnalysisR/reference/spacy_nlp.md),
[`spacy_parse_full()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_parse_full.md),
[`spacy_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_similarity.md)

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c("Apple was founded by Steve Jobs in Cupertino, California.")
entities <- spacy_extract_entities(texts)
print(entities)
} # }
```
