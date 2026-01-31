# Describe Image with Ollama Vision Model

Describe Image with Ollama Vision Model

## Usage

``` r
describe_image_ollama(
  image_base64,
  prompt =
    "Describe this image in detail, focusing on any charts, diagrams, tables, or textual content. Extract any visible text.",
  model = "llava",
  timeout = 120
)
```

## Arguments

- image_base64:

  Character string of base64-encoded PNG image

- prompt:

  Character string describing what to extract

- model:

  Character string, Ollama vision model name (default: "llava")

- timeout:

  Numeric, request timeout in seconds (default: 120)

## Value

Character string description, or NULL on failure
