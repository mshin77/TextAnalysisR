# Smart PDF Extraction with Auto-Detection

Extracts text and visual content from PDFs using R-native pdftools and
vision LLM APIs. Routes directly to multimodal extraction.

## Usage

``` r
extract_pdf_smart(
  file_path,
  doc_type = "auto",
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  envname = "textanalysisr-env"
)
```

## Arguments

- file_path:

  Character string path to PDF file

- doc_type:

  Character: "auto" (default), "academic", or "general" (kept for
  compatibility)

- vision_provider:

  Character: "ollama" (default), "openai", or "gemini"

- vision_model:

  Character: Model name for vision analysis

- api_key:

  Character: API key for cloud providers

- envname:

  Character: Kept for backward compatibility, ignored

## Value

List with extracted content ready for text analysis

## See also

Other pdf:
[`check_vision_models()`](https://mshin77.github.io/TextAnalysisR/reference/check_vision_models.md),
[`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md),
[`detect_pdf_content_type_py()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type_py.md),
[`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md),
[`extract_tables_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md),
[`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md),
[`extract_text_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf_py.md),
[`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md),
[`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)

## Examples

``` r
if (FALSE) { # \dontrun{
result <- extract_pdf_smart("document.pdf")
corpus <- prep_texts(result$combined_text)
} # }
```
