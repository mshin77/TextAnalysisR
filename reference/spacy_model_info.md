# Get spaCy Model Information

Get information about the loaded spaCy model.

## Usage

``` r
spacy_model_info(model = "en_core_web_sm")
```

## Arguments

- model:

  Character; spaCy model to query.

## Value

A list with model metadata including:

- `model_name`: Model identifier

- `lang`: Language code

- `pipeline`: List of pipeline components

- `has_vectors`: Whether model includes word vectors

- `vector_dim`: Dimension of word vectors (if available)

## See also

Other nlp:
[`spacy_extract_entities()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_entities.md),
[`spacy_extract_noun_chunks()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_noun_chunks.md),
[`spacy_nlp`](https://mshin77.github.io/TextAnalysisR/reference/spacy_nlp.md),
[`spacy_parse_full()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_parse_full.md),
[`spacy_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_similarity.md)

## Examples

``` r
if (FALSE) { # \dontrun{
info <- spacy_model_info()
print(info)
} # }
```
