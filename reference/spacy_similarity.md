# Calculate Text Similarity

Calculate semantic similarity between two texts using spaCy word
vectors. Requires a model with word vectors (en_core_web_md or
en_core_web_lg).

## Usage

``` r
spacy_similarity(text1, text2, model = "en_core_web_md")
```

## Arguments

- text1:

  First text string.

- text2:

  Second text string.

- model:

  Character; spaCy model to use (must have word vectors).

## Value

Numeric similarity score between 0 and 1.

## See also

Other nlp:
[`spacy_extract_entities()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_entities.md),
[`spacy_extract_noun_chunks()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_noun_chunks.md),
[`spacy_model_info()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_model_info.md),
[`spacy_nlp`](https://mshin77.github.io/TextAnalysisR/reference/spacy_nlp.md),
[`spacy_parse_full()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_parse_full.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Requires medium or large model
init_spacy_nlp("en_core_web_md")
sim <- spacy_similarity("I love dogs", "I adore puppies")
print(sim)  # High similarity

sim2 <- spacy_similarity("I love dogs", "The weather is nice")
print(sim2)  # Low similarity
} # }
```
