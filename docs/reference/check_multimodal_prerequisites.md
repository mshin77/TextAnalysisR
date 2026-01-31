# Check Multimodal Prerequisites

Checks all prerequisites for multimodal PDF extraction. Uses R-native
pdftools for rendering (no Python required).

## Usage

``` r
check_multimodal_prerequisites(
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  envname = "textanalysisr-env"
)
```

## Arguments

- vision_provider:

  Character: "ollama", "openai", or "gemini"

- vision_model:

  Character: Model name (optional)

- api_key:

  Character: API key for OpenAI/Gemini (if using cloud provider)

- envname:

  Character: Kept for backward compatibility, ignored

## Value

List with:

- ready: Logical - TRUE if all prerequisites met

- missing: Character vector of missing components

- instructions: Character - Detailed setup instructions

- details: List with component-specific status
