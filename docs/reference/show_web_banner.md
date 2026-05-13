# Show Web Deployment Banner

Show Web Deployment Banner

## Usage

``` r
show_web_banner(disabled = NULL)
```

## Arguments

- disabled:

  Optional character vector naming features to mark as disabled in the
  banner; `NULL` (default) shows the standard set.

## Value

A `shiny.tag` object containing the banner HTML for inclusion in a Shiny
UI, or `NULL` when not running in a web deployment context.
