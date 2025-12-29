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
  "textanalysisr-env")

## Value

Character string: "tabular", "text", or "unknown"

## See also

Other pdf:
[`check_vision_models()`](https://mshin77.github.io/TextAnalysisR/reference/check_vision_models.md),
[`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md),
[`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md),
[`extract_pdf_smart()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_smart.md),
[`extract_tables_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md),
[`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md),
[`extract_text_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf_py.md),
[`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md),
[`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)

## Examples

``` r
if (FALSE) { # \dontrun{
setup_python_env()

pdf_path <- "path/to/document.pdf"
content_type <- detect_pdf_content_type_py(pdf_path)
print(content_type)
} # }
```
