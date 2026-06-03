# Check Vision Model Availability

Check if required vision models are available for multimodal processing.

## Usage

``` r
check_vision_models(provider = "gemini", api_key = NULL)
```

## Arguments

- provider:

  Character: "openai" or "gemini"

- api_key:

  Character: API key (for OpenAI/Gemini)

## Value

List with availability status and recommendations

## Examples

``` r
# \donttest{
status <- check_vision_models("openai", api_key = Sys.getenv("OPENAI_API_KEY"))
status <- check_vision_models("gemini", api_key = Sys.getenv("GEMINI_API_KEY"))
# }
```
