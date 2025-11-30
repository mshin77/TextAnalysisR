# Call Ollama for Text Generation

Sends a prompt to Ollama and returns the generated text.

## Usage

``` r
call_ollama(
  prompt,
  model = "phi3:mini",
  temperature = 0.3,
  max_tokens = 512,
  timeout = 60,
  verbose = FALSE
)
```

## Arguments

- prompt:

  Character string containing the prompt.

- model:

  Character string specifying the Ollama model (default: "phi3:mini").

- temperature:

  Numeric value controlling randomness (default: 0.3).

- max_tokens:

  Maximum number of tokens to generate (default: 512).

- timeout:

  Timeout in seconds for the request (default: 60).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

Character string with the generated text, or NULL if failed.

## Examples

``` r
if (FALSE) { # \dontrun{
response <- call_ollama(
  prompt = "Summarize these keywords: machine learning, neural networks, AI",
  model = "phi3:mini"
)
print(response)
} # }
```
