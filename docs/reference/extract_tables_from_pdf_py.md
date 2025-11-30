# Extract Tables from PDF using Python

Extracts tabular data from PDF using pdfplumber (Python). No Java
required - pure Python solution.

## Usage

``` r
extract_tables_from_pdf_py(
  file_path,
  pages = NULL,
  envname = "textanalysisr-env"
)
```

## Arguments

- file_path:

  Character string path to PDF file

- pages:

  Integer vector of page numbers to process (NULL = all pages)

- envname:

  Character string, name of Python virtual environment (default:
  "langgraph-env")

## Value

Data frame with extracted table data Returns NULL if no tables found or
extraction fails

## Details

Uses pdfplumber Python library through reticulate. Works with complex
table layouts without Java dependency.

## Examples

``` r
if (FALSE) { # \dontrun{
setup_langgraph_env()

pdf_path <- "path/to/table_document.pdf"
table_data <- extract_tables_from_pdf_py(pdf_path)
head(table_data)
} # }
```
