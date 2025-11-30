# Get Recommended Ollama Model

Returns a recommended Ollama model based on what's available.

## Usage

``` r
get_recommended_ollama_model(
  preferred_models = c("phi3:mini", "llama3.1:8b", "mistral:7b", "tinyllama"),
  verbose = FALSE
)
```

## Arguments

- preferred_models:

  Character vector of preferred models in priority order.

- verbose:

  Logical, if TRUE, prints status messages.

## Value

Character string of recommended model, or NULL if none available.

## Examples

``` r
if (FALSE) { # \dontrun{
model <- get_recommended_ollama_model()
print(model)
} # }
```
