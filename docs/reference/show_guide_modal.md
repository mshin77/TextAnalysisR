# Show Guide Modal Dialog from HTML File

Loads and displays a modal dialog with guide content from an HTML file.
This function is designed for Shiny applications to display help
documentation stored in external HTML files, reducing server.R file size
and improving maintainability.

## Usage

``` r
show_guide_modal(guide_name, title, size = "l")
```

## Arguments

- guide_name:

  Name of the guide file (without .html extension). Files should be
  located in inst/TextAnalysisR.app/markdown/guides/

- title:

  Modal dialog title to display

- size:

  Size of the modal dialog (default: "l" for large). Options: "s"
  (small), "m" (medium), "l" (large)

## Value

Displays a Shiny modal dialog. Returns NULL invisibly.

## Details

Guide HTML files should be placed in:
`inst/TextAnalysisR.app/markdown/guides/<guide_name>.html`

The function will look for the guide file in the installed package
location. If the file is not found, it displays an error message in the
modal.

## Examples

``` r
if (FALSE) { # \dontrun{
observeEvent(input$showDimRedInfo, {
  show_guide_modal("dimensionality_reduction_guide", "Dimensionality Reduction Guide")
})

observeEvent(input$showClusteringInfo, {
  show_guide_modal("clustering_guide", "Document Clustering Guide")
})
} # }
```
