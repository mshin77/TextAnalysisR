# Plot Topic Effects for Continuous Variables

Creates a faceted plot showing how continuous variables affect topic
proportions.

## Usage

``` r
plot_topic_effects_continuous(
  effects_data,
  ncol = 2,
  height = 800,
  width = 1000,
  title = "Continuous Variable Effects",
  base_font_size = 11
)
```

## Arguments

- effects_data:

  Data frame with columns: topic, value, proportion, lower, upper

- ncol:

  Number of columns for faceting (default: 2)

- height:

  Plot height in pixels (default: 800)

- width:

  Plot width in pixels (default: 1000)

- title:

  Plot title (default: "Continuous Variable Effects")

- base_font_size:

  Base font size in points for the plot theme (default: 11). Axis text
  and strip text will be base_font_size + 2.

## Value

A plotly object
