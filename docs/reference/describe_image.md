# Describe Image Using Vision LLM

Unified dispatcher for image description using vision LLMs. Routes to
the appropriate provider (Ollama, OpenAI, or Gemini).

## Usage

``` r
describe_image(
  image_base64,
  provider = "ollama",
  model = NULL,
  api_key = NULL,
  prompt =
    "Describe this image: charts, diagrams, tables, and text. Extract visible text.",
  timeout = 120
)
```

## Arguments

- image_base64:

  Character string of base64-encoded PNG image

- provider:

  Character: "ollama", "openai", or "gemini"

- model:

  Character: Model name (uses provider default if NULL)

- api_key:

  Character: API key (required for openai/gemini)

- prompt:

  Character: Description prompt

- timeout:

  Numeric: Request timeout in seconds (default: 120)

## Value

Character string description, or NULL on failure
