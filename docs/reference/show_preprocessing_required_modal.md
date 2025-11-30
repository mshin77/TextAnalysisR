# Show Generic Preprocessing Required Modal

Displays a simple modal indicating preprocessing is required.
Lightweight alternative when detailed steps aren't needed.

## Usage

``` r
show_preprocessing_required_modal(
  message = "Please complete preprocessing steps first.",
  title = "Preprocessing Required"
)
```

## Arguments

- message:

  Custom message (default: "Please complete preprocessing steps first.")

- title:

  Modal title (default: "Preprocessing Required")

## Value

Displays a Shiny modal dialog. Returns NULL invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
if (!preprocessing_complete()) {
  show_preprocessing_required_modal()
  return()
}
} # }
```
