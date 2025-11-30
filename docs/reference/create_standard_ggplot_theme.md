# Create Standard ggplot2 Theme

Returns a standardized ggplot2 theme matching TextAnalysisR design
standards.

## Usage

``` r
create_standard_ggplot_theme(base_size = 14)
```

## Arguments

- base_size:

  Base font size (default: 14)

## Value

A ggplot2 theme object

## Examples

``` r
if (FALSE) { # \dontrun{
library(ggplot2)
ggplot(mtcars, aes(mpg, wt)) +
  geom_point() +
  create_standard_ggplot_theme()
} # }
```
