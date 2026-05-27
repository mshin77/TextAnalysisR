# Lemmatize Texts with spaCy

Perform lemmatization using spaCy with optimized pipeline settings.
Disables unnecessary components (NER, parser) for faster processing.

## Usage

``` r
spacy_lemmatize(x, batch_size = 100, model = "en_core_web_sm")
```

## Arguments

- x:

  Character vector of texts OR a quanteda tokens object.

- batch_size:

  Integer; batch size for processing (default: 100).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with columns: doc_id, token_id, token, lemma.

## Details

This function disables NER, entity_ruler, and parser components to speed
up lemmatization. Use this for lemmas without other annotations.
