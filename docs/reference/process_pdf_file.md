# Process PDF File

Main function to process PDF files - extracts text content using
pdftools. For table extraction, use
[`process_pdf_file_py`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md).

## Usage

``` r
process_pdf_file(file_path, content_type = "auto")
```

## Arguments

- file_path:

  Character string path to PDF file

- content_type:

  Character string: "auto" or "text" (default: "auto")

## Value

List with:

- data: Data frame with extracted content

- type: Character string indicating content type ("text" or "error")

- success: Logical indicating success

- message: Character string with status message

## Details

This function extracts text content from PDFs using pdftools package.
Works best with text-based PDFs (not scanned images).

For PDFs containing tables or complex layouts, use the Python-based
[`process_pdf_file_py`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)
which provides better table extraction.

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
[`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)

## Examples

``` r
if (FALSE) { # \dontrun{
pdf_path <- "path/to/document.pdf"
result <- process_pdf_file(pdf_path)

if (result$success) {
  print(head(result$data))
} else {
  print(result$message)
}
} # }
```
