# Extract PDF with Multimodal Analysis

Extract both text and visual content from PDFs using R-native pdftools
and vision LLM APIs. No Python required.

## Usage

``` r
extract_pdf_multimodal(
  file_path,
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  describe_images = TRUE,
  envname = "textanalysisr-env"
)
```

## Arguments

- file_path:

  Character string path to PDF file

- vision_provider:

  Character: "ollama" (local, default), "openai", or "gemini"

- vision_model:

  Character: Model name

  - For Ollama: "llava", "llava:13b", "bakllava"

  - For OpenAI: "gpt-4.1", "gpt-4.1-mini"

  - For Gemini: "gemini-2.5-flash", "gemini-2.5-pro"

- api_key:

  Character: API key (required for openai/gemini providers)

- describe_images:

  Logical: Convert page images to text descriptions (default: TRUE)

- envname:

  Character: Kept for backward compatibility, ignored

## Value

List with:

- success: Logical

- combined_text: Character string with all content for text analysis

- text_content: List of text chunks

- image_descriptions: List of image descriptions

- num_images: Integer count of described pages

- vision_provider: Character indicating provider used

- message: Character status message

## Details

**Workflow:**

1.  Extracts text using pdftools (R-native)

2.  Renders each page as an image

3.  Sends sparse-text pages to vision LLM for description

4.  Merges text + descriptions into a single text corpus

## See also

Other pdf:
[`check_vision_models()`](https://mshin77.github.io/TextAnalysisR/reference/check_vision_models.md),
[`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md),
[`detect_pdf_content_type_py()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type_py.md),
[`extract_pdf_smart()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_smart.md),
[`extract_tables_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md),
[`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md),
[`extract_text_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf_py.md),
[`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md),
[`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)

## Examples

``` r
if (FALSE) { # \dontrun{
result <- extract_pdf_multimodal("research_paper.pdf")
text_for_analysis <- result$combined_text

result <- extract_pdf_multimodal(
  "paper.pdf",
  vision_provider = "gemini",
  api_key = Sys.getenv("GEMINI_API_KEY")
)
} # }
```
