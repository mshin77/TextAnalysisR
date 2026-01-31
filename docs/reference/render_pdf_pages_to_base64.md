# Render PDF Pages to Base64 PNG

Renders each page of a PDF as a PNG image and returns base64-encoded
strings. Uses pdftools (R-native, no Python required).

## Usage

``` r
render_pdf_pages_to_base64(file_path, dpi = 150)
```

## Arguments

- file_path:

  Character string path to PDF file

- dpi:

  Numeric, rendering resolution (default: 150)

## Value

List of base64-encoded PNG strings, one per page
