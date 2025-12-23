# Process PDF File using Python

Main function to process PDF files using pdfplumber (Python).
Automatically detects content type and extracts data accordingly. No
Java required.

## Usage

``` r
process_pdf_file_py(
  file_path,
  content_type = "auto",
  envname = "textanalysisr-env"
)
```

## Arguments

- file_path:

  Character string path to PDF file

- content_type:

  Character string: "auto", "text", or "tabular" If "auto", will detect
  content type automatically

- envname:

  Character string, name of Python virtual environment (default:
  "langgraph-env")

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

- Uses same Python environment as LangGraph

## See also

Other pdf:
[`check_vision_models()`](https://mshin77.github.io/TextAnalysisR/reference/check_vision_models.md),
[`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md),
[`detect_pdf_content_type_py()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type_py.md),
[`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md),
[`extract_pdf_smart()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_smart.md),
[`extract_tables_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md),
[`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md),
[`extract_text_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf_py.md),
[`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md)

## Examples

``` r
if (FALSE) { # \dontrun{
setup_langgraph_env()

pdf_path <- "path/to/document.pdf"
result <- process_pdf_file_py(pdf_path)

if (result$success) {
  print(head(result$data))
} else {
  print(result$message)
}
} # }
```
