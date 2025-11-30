# Detect PDF Content Type using Python

Analyzes PDF to determine if it contains primarily tabular data or text.

## Usage

``` r
detect_pdf_content_type_py(file_path, envname = "textanalysisr-env")
```

## Arguments

- file_path:

  Character string path to PDF file

- envname:

  Character string, name of Python virtual environment (default:
  "langgraph-env")

## Value

Character string: "tabular", "text", or "unknown"

## Examples

``` r
if (FALSE) { # \dontrun{
setup_langgraph_env()

pdf_path <- "path/to/document.pdf"
content_type <- detect_pdf_content_type_py(pdf_path)
print(content_type)
} # }
```
