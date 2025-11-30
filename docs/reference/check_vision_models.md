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
