# Parse Morphology String into Individual Columns

Parse spaCy's morphology string format (e.g.,
"Number=Sing\|Tense=Past\|VerbForm=Fin") into individual columns.

## Usage

``` r
parse_morphology_string(parsed, features)
```

## Arguments

- parsed:

  Data frame with a 'morph' column from spaCy.

- features:

  Character vector of feature names to extract.

## Value

Data frame with additional morph\_\* columns for each feature.
