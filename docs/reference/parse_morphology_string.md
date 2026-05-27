# Parse Morphology String

Parse spaCy's morphology string format into individual columns. Used
internally by morphology analysis functions. Always extracts all common
morphology features (Number, Tense, VerbForm, Person, Case, Mood,
Aspect) regardless of the features parameter.

## Usage

``` r
parse_morphology_string(data, features = NULL)
```

## Arguments

- data:

  Data frame with a 'morph' column from spaCy parsing.

- features:

  Character vector of feature names (ignored, kept for backwards
  compatibility). All features are always extracted.

## Value

Data frame with additional morph\_\* columns for each feature.
