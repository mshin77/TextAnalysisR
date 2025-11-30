# Check Alt Text Presence

Validates that images and visualizations have alternative text
descriptions. Required for WCAG 1.1.1 (Non-text Content).

Note: Decorative images should use empty alt text (alt="") to indicate
they should be ignored by assistive technology.

## Usage

``` r
check_alt_text(alt_text, element_type = "image", decorative = FALSE)
```

## Arguments

- alt_text:

  Alternative text description

- element_type:

  Type of element (e.g., "plot", "image", "icon")

- decorative:

  Logical, TRUE if element is purely decorative

## Value

Logical TRUE if valid, FALSE with warning if missing/inadequate

## Examples

``` r
if (FALSE) { # \dontrun{
check_alt_text("Bar chart showing word frequency", "plot")  # TRUE
check_alt_text("", "plot")  # FALSE (informative content needs alt text)
check_alt_text("", "icon", decorative = TRUE)  # TRUE (decorative is OK)
} # }
```
