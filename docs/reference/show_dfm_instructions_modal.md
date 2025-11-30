# Show DFM Setup Instructions Modal

Displays a modal dialog with console-style instructions for creating a
DFM. Uses verbatimTextOutput for formatting.

## Usage

``` r
show_dfm_instructions_modal(
  output_id,
  feature_name = "this feature",
  session = NULL
)
```

## Arguments

- output_id:

  Shiny output ID for the verbatimTextOutput

- feature_name:

  Name of the feature requiring DFM (default: "this feature")

- session:

  Shiny session object (default: getDefaultReactiveDomain())

## Value

Displays a Shiny modal dialog. Returns NULL invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
output$dfm_instructions <- renderPrint({
  cat(get_dfm_setup_instructions("keywords"), sep = "\n")
})

show_dfm_instructions_modal("dfm_instructions", "keywords")
} # }
```
