# Show Preprocessing Steps Modal

Displays a modal dialog listing required preprocessing steps for a
feature. Generic version that works for any feature requiring
preprocessing.

## Usage

``` r
show_preprocessing_steps_modal(
  title = "Preprocessing Required",
  message,
  required_steps,
  optional_steps = NULL,
  additional_note = NULL
)
```

## Arguments

- title:

  Modal title (default: "Preprocessing Required")

- message:

  Main message to display

- required_steps:

  Character vector of required preprocessing steps

- optional_steps:

  Character vector of optional preprocessing steps (default: NULL)

- additional_note:

  Optional additional note to display (default: NULL)

## Value

Displays a Shiny modal dialog. Returns NULL invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
show_preprocessing_steps_modal(
  message = "Please complete preprocessing to generate tokens.",
  required_steps = c("Step 1: Unite Texts", "Step 4: Document-Feature Matrix"),
  optional_steps = c("Steps 2, 3, 5, and 6")
)
} # }
```
