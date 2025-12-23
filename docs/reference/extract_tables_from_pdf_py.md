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

## See also

Other pdf:
[`check_vision_models()`](https://mshin77.github.io/TextAnalysisR/reference/check_vision_models.md),
[`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md),
[`detect_pdf_content_type_py()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type_py.md),
[`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md),
[`extract_pdf_smart()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_smart.md),
[`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md),
[`extract_text_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf_py.md),
[`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md),
[`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)

## Examples

``` r
if (FALSE) { # \dontrun{
setup_langgraph_env()

pdf_path <- "path/to/table_document.pdf"
table_data <- extract_tables_from_pdf_py(pdf_path)
head(table_data)
} # }
```
