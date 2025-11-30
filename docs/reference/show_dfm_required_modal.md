# Show DFM Requirement Modal

Displays a standardized modal dialog informing users they need to
complete preprocessing steps before using a feature that requires a
document-feature matrix.

## Usage

``` r
show_dfm_required_modal(
  feature_name = "this feature",
  additional_message = NULL
)
```

## Arguments

- feature_name:

  Name of the feature requiring DFM (e.g., "topic modeling", "keyword
  extraction")

- additional_message:

  Optional additional message to display (default: NULL)

## Value

Displays a Shiny modal dialog. Returns NULL invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
if (is.null(dfm_init())) {
  show_dfm_required_modal("topic modeling")
  return(NULL)
}
} # }
```
