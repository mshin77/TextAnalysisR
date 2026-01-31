# Process PDF File (Unified Entry Point)

Unified PDF processing:

1.  Multimodal (R-native pdftools + Vision LLM) if enabled

2.  R pdftools text extraction as fallback

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

  Character, "ollama", "openai", or "gemini"

- vision_model:

  Character, model name

- api_key:

  Character, API key (if using openai/gemini)

- describe_images:

  Logical, generate image descriptions

## Value

List: success, data, type, method, message

## See also

Other preprocessing:
[`get_available_dfm()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_dfm.md),
[`get_available_tokens()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_tokens.md),
[`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md),
[`lemmatize_tokens()`](https://mshin77.github.io/TextAnalysisR/reference/lemmatize_tokens.md),
[`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md),
[`unite_cols()`](https://mshin77.github.io/TextAnalysisR/reference/unite_cols.md)
