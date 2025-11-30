# Process PDF File (Unified Entry Point)

Unified PDF processing with automatic fallback:

1.  Multimodal (Python + Vision) 2. Python pdfplumber 3. R pdftools

## Usage

``` r
process_pdf_unified(
  file_path,
  use_multimodal = FALSE,
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  describe_images = TRUE
)
```

## Arguments

- file_path:

  Character string path to PDF file

- use_multimodal:

  Logical, enable multimodal extraction

- vision_provider:

  Character, "ollama" or "openai"

- vision_model:

  Character, model name

- api_key:

  Character, OpenAI API key (if using OpenAI)

- describe_images:

  Logical, generate image descriptions

## Value

List: success, data, type, method, message
