# Check Vision Model Availability

Check if required vision models are available for multimodal processing.

## Usage

``` r
check_vision_models(provider = "ollama", api_key = NULL)
```

## Arguments

- provider:

  Character: "ollama", "openai", or "gemini"

- api_key:

  Character: API key (for OpenAI/Gemini)

## Value

List with availability status and recommendations

## Examples

``` r
if (interactive()) {
status <- check_vision_models("ollama")
status <- check_vision_models("gemini", api_key = Sys.getenv("GEMINI_API_KEY"))
}
```
