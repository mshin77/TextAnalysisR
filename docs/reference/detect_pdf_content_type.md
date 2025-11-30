# Detect PDF Content Type

Analyzes PDF to determine if it contains readable text.

## Usage

``` r
detect_pdf_content_type(file_path)
```

## Arguments

- file_path:

  Character string path to PDF file

## Value

Character string: "text" or "unknown"

## Details

Attempts text extraction using pdftools. Returns "text" if successful,
or "unknown" if extraction fails or PDF is empty.

For table extraction from PDFs, use
[`extract_tables_from_pdf_py`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md).

## Examples

``` r
if (FALSE) { # \dontrun{
pdf_path <- "path/to/document.pdf"
content_type <- detect_pdf_content_type(pdf_path)
print(content_type)
} # }
```
