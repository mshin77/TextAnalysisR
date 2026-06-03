# Extract Text from PDF using Python

Extracts text content from a PDF file using pdfplumber (Python). No Java
required - uses Python environment.

## Usage

``` r
extract_text_from_pdf_py(file_path, envname = NULL)
```

## Arguments

- file_path:

  Character string path to PDF file

- envname:

  Character string, name of Python virtual environment (default:
  "textanalysisr-env")

## Value

Data frame with columns: page (integer), text (character) Returns NULL
if extraction fails or PDF is empty

## Details

Uses pdfplumber Python library through reticulate. Requires Python
environment setup. See
[`setup_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/setup_python_env.md).
