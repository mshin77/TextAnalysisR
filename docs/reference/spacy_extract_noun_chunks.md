# Extract Noun Chunks

Extract noun chunks (base noun phrases) using spaCy.

## Usage

``` r
spacy_extract_noun_chunks(texts, model = "en_core_web_sm")
```

## Arguments

- texts:

  Character vector of texts.

- model:

  Character; spaCy model to use.

## Value

A data.frame with noun chunk information.

## See also

Other nlp:
[`spacy_extract_entities()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_entities.md),
[`spacy_model_info()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_model_info.md),
[`spacy_nlp`](https://mshin77.github.io/TextAnalysisR/reference/spacy_nlp.md),
[`spacy_parse_full()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_parse_full.md),
[`spacy_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_similarity.md)

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c("The quick brown fox jumps over the lazy dog.")
chunks <- spacy_extract_noun_chunks(texts)
print(chunks)
} # }
```
