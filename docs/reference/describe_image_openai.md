# Describe Image with OpenAI Vision API

Describe Image with OpenAI Vision API

## Usage

``` r
describe_image_openai(
  image_base64,
  prompt =
    "Describe this image in detail, focusing on any charts, diagrams, tables, or textual content. Extract any visible text.",
  model = "gpt-4.1",
  max_tokens = 500,
  api_key
)
```

## Arguments

- image_base64:

  Character string of base64-encoded PNG image

- prompt:

  Character string describing what to extract

- model:

  Character string, OpenAI model name (default: "gpt-4.1")

- max_tokens:

  Integer, maximum tokens in response (default: 500)

- api_key:

  Character string, OpenAI API key

## Value

Character string description, or NULL on failure
