# Get Recommended Ollama Model

Returns a recommended Ollama model based on what's available.

## Usage

``` r
get_recommended_ollama_model(
  preferred_models = c("llama3.2", "gemma3", "mistral:7b", "gemma3:1b", "tinyllama"),
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
if (interactive()) {
model <- get_recommended_ollama_model()
print(model)
}
```
