# Check Vision Model Availability

Check if required vision models are available for multimodal processing.

## Usage

``` r
check_vision_models(provider = "ollama", api_key = NULL)
```

## Arguments

- provider:

  Character: "ollama" or "openai"

- api_key:

  Character: API key (for OpenAI)

## Value

List with availability status and recommendations

## See also

Other pdf:
[`detect_pdf_content_type()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type.md),
[`detect_pdf_content_type_py()`](https://mshin77.github.io/TextAnalysisR/reference/detect_pdf_content_type_py.md),
[`extract_pdf_multimodal()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_multimodal.md),
[`extract_pdf_smart()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pdf_smart.md),
[`extract_tables_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_tables_from_pdf_py.md),
[`extract_text_from_pdf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf.md),
[`extract_text_from_pdf_py()`](https://mshin77.github.io/TextAnalysisR/reference/extract_text_from_pdf_py.md),
[`process_pdf_file()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file.md),
[`process_pdf_file_py()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_file_py.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Check Ollama vision models
status <- check_vision_models("ollama")
print(status$message)

# Check OpenAI access
status <- check_vision_models("openai", api_key = Sys.getenv("OPENAI_API_KEY"))
} # }
```
