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
  title = "Continuous Variable Effects"
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

## Value

A plotly object
