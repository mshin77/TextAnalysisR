# Extract Text from PDF

Extracts text content from a PDF file using pdftools package.

## Usage

``` r
extract_text_from_pdf(file_path)
```

## Arguments

- file_path:

  Character string path to PDF file

## Value

Data frame with columns: page (integer), text (character) Returns NULL
if extraction fails or PDF is empty

## Details

Uses
[`pdftools::pdf_text()`](https://docs.ropensci.org/pdftools//reference/pdftools.html)
to extract text from each page. Preserves page structure and cleans
whitespace. Works best with text-based PDFs (not scanned images).

## See also

Other pdf:
[`check_vision_models()`](https://mshin77.github.io/TextAnalysisR/reference/check_vision_models.md),
[`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md),
[`detect_pdf_content_type_py()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type_py.md),
[`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md),
[`extract_pdf_smart()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_smart.md),
[`extract_tables_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md),
[`extract_text_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf_py.md),
[`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md),
[`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)

## Examples

``` r
if (FALSE) { # \dontrun{
pdf_path <- "path/to/document.pdf"
text_data <- extract_text_from_pdf(pdf_path)
head(text_data)
} # }
```
