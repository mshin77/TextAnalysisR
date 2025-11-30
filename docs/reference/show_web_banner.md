# Show Web Deployment Banner

Creates a Shiny UI banner for web deployments showing feature
limitations.

## Usage

``` r
show_web_banner(disabled = NULL)
```

## Arguments

- disabled:

  Character vector of disabled feature names (optional)

## Value

A shiny tagList UI element (or NULL if local)

## Examples

``` r
if (FALSE) { # \dontrun{
output$banner <- renderUI({ show_web_banner() })
} # }
```
