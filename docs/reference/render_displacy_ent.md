# Render displaCy Entity Visualization

Renders spaCy's displaCy entity visualization as HTML. Highlights named
entities with colored labels.

## Usage

``` r
render_displacy_ent(text, model = "en_core_web_sm", colors = NULL)
```

## Arguments

- text:

  Character string to visualize.

- model:

  spaCy model name (default: "en_core_web_sm").

- colors:

  Named list of entity type to color mappings (e.g., list(PERSON =
  "#e91e63", ORG = "#2196f3")). If NULL, uses spaCy defaults.

## Value

HTML string with entity highlighting.
