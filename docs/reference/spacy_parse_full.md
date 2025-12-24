# Parse Texts with Full spaCy Features

Parse texts using spaCy and extract all linguistic features including
morphology, dependency parsing, POS tags, lemmas, and named entities.

## Usage

``` r
spacy_parse_full(
  texts,
  include_pos = TRUE,
  include_lemma = TRUE,
  include_entity = TRUE,
  include_dependency = TRUE,
  include_morphology = TRUE,
  model = "en_core_web_sm"
)
```

## Arguments

- texts:

  Character vector of texts to parse.

- include_pos:

  Logical; include part-of-speech tags (default: TRUE).

- include_lemma:

  Logical; include lemmatized forms (default: TRUE).

- include_entity:

  Logical; include named entity recognition (default: TRUE).

- include_dependency:

  Logical; include dependency parsing (default: TRUE).

- include_morphology:

  Logical; include morphological features (default: TRUE).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data.frame with token-level annotations:

- `doc_id`: Document identifier

- `sentence_id`: Sentence number within document

- `token_id`: Token position within sentence

- `token`: Original token text

- `pos`: Universal POS tag (NOUN, VERB, ADJ, etc.)

- `tag`: Fine-grained POS tag (NN, VBD, JJ, etc.)

- `lemma`: Lemmatized form

- `entity`: Named entity type (PERSON, ORG, GPE, etc.)

- `entity_iob`: IOB tag (B=beginning, I=inside, O=outside)

- `dep_rel`: Dependency relation (nsubj, dobj, amod, etc.)

- `head_token_id`: Head token in dependency tree

- `morph`: Full morphological features string

- `morph_*`: Individual morphological features as columns

## Details

This function uses Python spaCy via reticulate, providing access to
features not available in spacyr, including morphological analysis.

Morphological features include:

- `morph_Number`: Sing, Plur

- `morph_Person`: 1, 2, 3

- `morph_Tense`: Past, Pres, Fut

- `morph_VerbForm`: Fin, Inf, Part, Ger

- `morph_Mood`: Ind, Imp, Sub

- `morph_Case`: Nom, Acc, Dat, Gen

- And more depending on the language model

## See also

Other nlp:
[`spacy_extract_entities()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_entities.md),
[`spacy_extract_noun_chunks()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_noun_chunks.md),
[`spacy_model_info()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_model_info.md),
[`spacy_nlp`](https://mshin77.github.io/TextAnalysisR/reference/spacy_nlp.md),
[`spacy_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_similarity.md)

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c(
  "Apple Inc. was founded by Steve Jobs.",
  "The cats are sleeping on the couch."
)
result <- spacy_parse_full(texts)

# View morphological features
result[, c("token", "pos", "morph")]

# Filter by POS
verbs <- result[result$pos == "VERB", ]
print(verbs[, c("token", "lemma", "morph_Tense", "morph_VerbForm")])
} # }
```
