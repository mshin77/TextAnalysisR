# Plot Named Entity Frequencies

Creates a bar plot showing the frequency distribution of named entity
types.

## Usage

``` r
plot_entity_frequencies(
  entity_data,
  top_n = 20,
  title = "Named Entity Type Frequency",
  color = NULL,
  height = 500,
  width = NULL,
  custom_colors = NULL
)
```

## Arguments

- entity_data:

  Data frame containing entity data with columns:

  - `entity`: Named entity type (e.g., "PERSON", "ORG", "GPE")

  - `n`: (optional) Pre-computed frequency count

  If `n` is not present, frequencies will be computed from the data.

- top_n:

  Number of top entity types to display (default: 20)

- title:

  Plot title (default: "Named Entity Type Frequency")

- color:

  Bar color (default: "#10B981")

- height:

  Plot height in pixels (default: 500)

- width:

  Plot width in pixels (default: NULL for auto)

- custom_colors:

  Named vector of custom entity type colors (e.g., c(CONCEPT =
  "#00acc1", THEME = "#7c4dff")). Custom colors override defaults.

## Value

A plotly object

## Examples

``` r
if (interactive()) {
  entity_df <- data.frame(
    entity = c("PERSON", "ORG", "GPE", "DATE", "MONEY"),
    n = c(300, 250, 200, 150, 100)
  )
  plot_entity_frequencies(entity_df)

  # With custom colors
  plot_entity_frequencies(entity_df, custom_colors = c(PERSON = "#ff0000"))
}
```
