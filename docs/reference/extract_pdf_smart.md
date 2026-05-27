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
