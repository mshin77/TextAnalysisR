# Check if Ollama is Available

Checks if Ollama is installed and running on the local machine.

## Usage

``` r
check_ollama(verbose = FALSE)
```

## Arguments

- verbose:

  Logical, if TRUE, prints status messages.

## Value

Logical indicating whether Ollama is available.

## Examples

``` r
if (FALSE) { # \dontrun{
if (check_ollama()) {
  message("Ollama is ready!")
}
} # }
```
