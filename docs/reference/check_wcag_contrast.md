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
