# Describe Image with Gemini Vision API

Describe Image with Gemini Vision API

## Usage

``` r
describe_image_gemini(
  image_base64,
  prompt =
    "Describe this image in detail, focusing on any charts, diagrams, tables, or textual content. Extract any visible text.",
  model = "gemini-2.5-flash",
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

  Character string, Gemini model name (default: "gemini-2.5-flash")

- max_tokens:

  Integer, maximum tokens in response (default: 500)

- api_key:

  Character string, Gemini API key

## Value

Character string description, or NULL on failure
