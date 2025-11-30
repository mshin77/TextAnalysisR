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
  envname = "langgraph-env"
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
