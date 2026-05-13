# Generate ARIA Label

Creates accessible ARIA label for UI elements.

## Usage

``` r
create_aria_label(element_type, action, context = NULL)
```

## Arguments

- element_type:

  Type of element (e.g., "button", "input", "plot")

- action:

  Action or purpose (e.g., "analyze", "download", "visualize")

- context:

  Additional context (optional)

## Value

Character string with ARIA label

## Examples

``` r
create_aria_label("button", "analyze", "readability")
#> [1] "Analyze readability button"
# Returns: "Analyze readability button"
```
