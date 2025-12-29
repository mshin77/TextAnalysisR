# Smart PDF Extraction with Auto-Detection

Automatically detects document type and chooses best extraction method:

- Academic papers → Nougat (equations)

- Documents with visuals → Multimodal extraction

- General documents → Marker

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

  Character: "auto" (default), "academic", or "general"

- vision_provider:

  Character: "ollama" (default) or "openai"

- vision_model:

  Character: Model name for vision analysis

- api_key:

  Character: API key for cloud providers

- envname:

  Character: Python environment name

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
# Auto-detect and extract
result <- extract_pdf_smart("document.pdf")

# Feed to text analysis
corpus <- prep_texts(result$combined_text)
topics <- fit_semantic_model(corpus, k = 10)

# Force academic extraction (with equations)
result <- extract_pdf_smart("paper.pdf", doc_type = "academic")
} # }
```
