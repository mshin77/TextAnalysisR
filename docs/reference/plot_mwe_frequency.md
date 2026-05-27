# Plot Multi-Word Expression Frequency

Creates a bar plot showing multi-word expression frequencies with
optional source-based coloring to distinguish between detected and
manually added expressions.

## Usage

``` r
plot_mwe_frequency(
  mwe_data,
  title = "Multi-Word Expression Frequency",
  color_by_source = TRUE,
  primary_color = "#10B981",
  secondary_color = "#A855F7",
  height = 500,
  width = NULL
)
```

## Arguments

- mwe_data:

  Data frame containing MWE data with columns:

  - `feature`: The multi-word expression text

  - `frequency`: Frequency count

  - `rank`: (optional) Rank of the expression

  - `docfreq`: (optional) Document frequency

  - `source`: (optional) Source category (e.g., "Top 20", "Manual")

- title:

  Plot title (default: "Multi-Word Expression Frequency")

- color_by_source:

  Whether to color bars by source column (default: TRUE)

- primary_color:

  Color for primary/top expressions (default: "#10B981")

- secondary_color:

  Color for secondary/manual expressions (default: "#A855F7")

- height:

  Plot height in pixels (default: 500)

- width:

  Plot width in pixels (default: NULL for auto)

## Value

A plotly object
