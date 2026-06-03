# Process PDF File using Python

Main function to process PDF files using pdfplumber (Python).
Automatically detects content type and extracts data accordingly. No
Java required.

## Usage

``` r
process_pdf_file_py(file_path, content_type = "auto", envname = NULL)
```

## Arguments

- file_path:

  Character string path to PDF file

- content_type:

  Character string: "auto", "text", or "tabular" If "auto", will detect
  content type automatically

- envname:

  Character string, name of Python virtual environment (default:
  "textanalysisr-env")

## Value

List with:

- data: Data frame with extracted content

- type: Character string indicating content type

- success: Logical indicating success

- message: Character string with status message

## Details

This function uses Python's pdfplumber library which:

- Handles both text and tables

- No Java dependency

- Better accuracy than tabulizer for complex tables

- Uses TextAnalysisR Python environment
