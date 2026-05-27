# Extract Morphological Features

Uses spaCy to extract morphological features from text. Returns data
with Number, Tense, VerbForm, Person, Case, Mood, Aspect, etc.

## Usage

``` r
extract_morphology(
  tokens,
  features = c("Number", "Tense", "VerbForm", "Person", "Case", "Mood", "Aspect"),
  include_pos = TRUE,
  include_lemma = TRUE,
  model = "en_core_web_sm"
)
```

## Arguments

- tokens:

  A quanteda tokens object or character vector of texts.

- features:

  Character vector of morphological features to extract. Default
  includes common Universal Dependencies features.

- include_pos:

  Logical; include POS tags (default: TRUE).

- include_lemma:

  Logical; include lemmatized forms (default: TRUE).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with token-level morphological annotations including
morph\_\* columns for each requested feature.

## Details

Morphological features follow Universal Dependencies annotation. Common
features include:

- `Number`: Sing (singular), Plur (plural)

- `Tense`: Past, Pres (present), Fut (future)

- `VerbForm`: Fin (finite), Inf (infinitive), Part (participle), Ger
  (gerund)

- `Person`: 1, 2, 3 (first, second, third person)

- `Case`: Nom (nominative), Acc (accusative), Gen (genitive), Dat
  (dative)

- `Mood`: Ind (indicative), Imp (imperative), Sub (subjunctive)

- `Aspect`: Perf (perfective), Imp (imperfective), Prog (progressive)

## Examples

``` r
if (interactive()) {
  tokens <- quanteda::tokens(TextAnalysisR::SpecialEduTech$abstract[1])
  morphology_data <- extract_morphology(tokens)
  print(morphology_data)
}
```
