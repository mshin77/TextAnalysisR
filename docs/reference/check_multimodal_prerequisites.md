# Check Multimodal Prerequisites

Checks all prerequisites for multimodal PDF extraction and returns
detailed status with setup instructions.

## Usage

``` r
check_multimodal_prerequisites(
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  envname = "langgraph-env"
)
```

## Arguments

- vision_provider:

  Character: "ollama" or "openai"

- vision_model:

  Character: Model name (optional)

- api_key:

  Character: API key for OpenAI (if using openai provider)

- envname:

  Character: Python environment name

## Value

List with:

- ready: Logical - TRUE if all prerequisites met

- missing: Character vector of missing components

- instructions: Character - Detailed setup instructions

- details: List with component-specific status
