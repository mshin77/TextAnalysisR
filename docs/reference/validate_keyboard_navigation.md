# Validate Keyboard Navigation

Checks if interactive elements have proper tabindex and keyboard
handlers. Used for WCAG 2.1.1 (Keyboard) compliance.

## Usage

``` r
validate_keyboard_navigation(tabindex = 0)
```

## Arguments

- tabindex:

  Integer, tab order (-1 for no tab, 0 for natural order, 1+ for
  specific order)

## Value

Logical TRUE if valid, FALSE with warning if invalid

## Examples

``` r
if (FALSE) { # \dontrun{
validate_keyboard_navigation(0)   # TRUE
validate_keyboard_navigation(999) # FALSE (too high)
} # }
```
