# Apply Standard Plotly Layout

Applies consistent layout styling to plotly plots following
TextAnalysisR design standards. This ensures all plots have uniform
fonts, colors, margins, and interactive features.

## Usage

``` r
apply_standard_plotly_layout(
  plot,
  title = NULL,
  xaxis_title = NULL,
  yaxis_title = NULL,
  margin = list(t = 60, b = 80, l = 80, r = 40),
  show_legend = FALSE
)
```

## Arguments

- plot:

  A plotly plot object

- title:

  Plot title text (optional)

- xaxis_title:

  X-axis title (optional)

- yaxis_title:

  Y-axis title (optional)

- margin:

  List of margins: list(t, b, l, r) in pixels (default: list(t = 60, b =
  80, l = 80, r = 40))

- show_legend:

  Logical, whether to show legend (default: FALSE)

## Value

A plotly plot object with standardized layout

## Details

Design standards applied:

- Title: 14px Roboto, \#0c1f4a

- Axis titles: 13px Roboto, \#0c1f4a

- Axis tick labels: 12px Roboto, \#3B3B3B
