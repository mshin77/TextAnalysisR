# Check WCAG Contrast Compliance

Validates if color combination meets WCAG 2.1 Level AA contrast
requirements.

## Usage

``` r
check_wcag_contrast(foreground, background, large_text = FALSE)
```

## Arguments

- foreground:

  Foreground color (hex format)

- background:

  Background color (hex format)

- large_text:

  Logical, TRUE if text is large (18pt+ or 14pt+ bold)

## Value

Logical TRUE if compliant, FALSE if not

## Examples

``` r
check_wcag_contrast("#111827", "#ffffff")  # TRUE (16:1 ratio)
#> [1] TRUE
check_wcag_contrast("#6b7280", "#4a5568")  # FALSE (2.8:1 ratio)
#> Warning: WCAG contrast failure: 1.56:1 ratio (requires 4.5:1)
#>   Foreground: #6b7280
#>   Background: #4a5568
#> [1] FALSE
```
