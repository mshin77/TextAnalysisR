# Call Ollama for Text Generation

Sends a prompt to Ollama and returns the generated text.

## Usage

``` r
call_ollama(
  prompt,
  model = "llama3.2",
  system = NULL,
  temperature = 0.3,
  max_tokens = 512,
  timeout = 120,
  verbose = FALSE
)
```

## Arguments

- prompt:

  Character string containing the prompt.

- model:

  Character string specifying the Ollama model (default: "llama3.2").

- system:

  Character string with system instructions (default: NULL).

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
