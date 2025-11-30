# List Available Ollama Models

Lists all models currently installed in Ollama.

## Usage

``` r
list_ollama_models(verbose = FALSE)
```

## Arguments

- verbose:

  Logical, if TRUE, prints status messages.

## Value

Character vector of model names, or NULL if Ollama is unavailable.

## Examples

``` r
if (FALSE) { # \dontrun{
models <- list_ollama_models()
print(models)
} # }
```
