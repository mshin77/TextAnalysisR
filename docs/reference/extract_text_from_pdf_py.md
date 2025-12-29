# Extract Text from PDF using Python

Extracts text content from a PDF file using pdfplumber (Python). No Java
required - uses Python environment.

## Usage

``` r
extract_text_from_pdf_py(file_path, envname = "textanalysisr-env")
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

## See also

Other pdf:
[`check_vision_models()`](https://mshin77.github.io/TextAnalysisR/reference/check_vision_models.md),
[`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md),
[`detect_pdf_content_type_py()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type_py.md),
[`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md),
[`extract_pdf_smart()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_smart.md),
[`extract_tables_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md),
[`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md),
[`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md),
[`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)

## Examples

``` r
if (FALSE) { # \dontrun{
setup_python_env()

pdf_path <- "path/to/document.pdf"
text_data <- extract_text_from_pdf_py(pdf_path)
head(text_data)
} # }
```
