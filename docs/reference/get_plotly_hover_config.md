# Get Standard Plotly Hover Label Configuration

Returns standardized hover label styling for plotly plots.

## Usage

``` r
get_plotly_hover_config(bgcolor = "#ffffff", fontcolor = "#0c1f4a")
```

## Arguments

- bgcolor:

  Background color (default: "#ffffff")

- fontcolor:

  Font color (default: "#0c1f4a")

## Value

A list of hover label configuration parameters

## Examples

``` r
if (FALSE) { # \dontrun{
hover_config <- get_plotly_hover_config()
plot_ly(..., hoverlabel = hover_config)
} # }
```
