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

- Title: 20px Roboto, \#0c1f4a

- Axis titles: 18px Roboto, \#0c1f4a

- Axis labels: 18px Roboto, \#3B3B3B

- Hover tooltips: 16px Roboto

- WCAG AA compliant colors

## Examples

``` r
if (FALSE) { # \dontrun{
library(plotly)
p <- plot_ly(x = 1:10, y = rnorm(10), type = "scatter", mode = "markers")
p %>% apply_standard_plotly_layout(
  title = "My Plot",
  xaxis_title = "X Values",
  yaxis_title = "Y Values"
)
} # }
```
