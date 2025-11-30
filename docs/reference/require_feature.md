# Require Feature

Checks feature availability and shows notification if unavailable.

## Usage

``` r
require_feature(feature, session = NULL)
```

## Arguments

- feature:

  Character: feature name to check

- session:

  Shiny session object (optional)

## Value

Logical TRUE if available, FALSE if not

## Examples

``` r
if (FALSE) { # \dontrun{
if (!require_feature("embeddings", session)) return()
} # }
```
