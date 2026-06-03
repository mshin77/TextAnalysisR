# Render displaCy Dependency Visualization

Renders spaCy's displaCy dependency visualization as SVG. Shows
syntactic structure with arrows between words.

## Usage

``` r
render_displacy_dep(text, compact = TRUE, model = "en_core_web_sm")
```

## Arguments

- text:

  Character string to visualize.

- compact:

  Logical; use compact mode for space (default: TRUE).

- model:

  spaCy model name (default: "en_core_web_sm").

## Value

SVG string with dependency tree.
