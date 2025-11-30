# Generate DFM Setup Instructions Text

Generates standardized text instructions for creating a DFM. Used in
console output or verbatim text displays.

## Usage

``` r
get_dfm_setup_instructions(feature_name = "this feature")
```

## Arguments

- feature_name:

  Name of the feature requiring DFM (default: "this feature")

## Value

Character vector of instruction lines

## Examples

``` r
if (FALSE) { # \dontrun{
output$instructions <- renderPrint({
  cat(get_dfm_setup_instructions("keyword extraction"), sep = "\n")
})
} # }
```
