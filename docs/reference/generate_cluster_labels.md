# Generate Cluster Labels with AI

Generates descriptive labels for clusters using either Ollama (local,
default) or OpenAI's API. When running locally, Ollama is preferred for
privacy and cost-free operation.

## Usage

``` r
generate_cluster_labels(
  cluster_keywords,
  provider = "auto",
  model = NULL,
  temperature = 0.3,
  max_tokens = 50,
  verbose = TRUE
)
```

## Arguments

- cluster_keywords:

  List of keywords for each cluster.

- provider:

  AI provider to use: "auto" (default), "ollama", or "openai". "auto"
  will use Ollama if available, otherwise OpenAI.

- model:

  Model name. For Ollama (default: "phi3:mini"). For OpenAI (default:
  "gpt-3.5-turbo").

- temperature:

  Temperature parameter (default: 0.3).

- max_tokens:

  Maximum tokens for response (default: 50).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A list of generated labels.

## Examples

``` r
if (FALSE) { # \dontrun{
keywords <- list("1" = c("machine", "learning", "neural"), "2" = c("data", "analysis"))
labels_ollama <- generate_cluster_labels(keywords, provider = "ollama")
labels_openai <- generate_cluster_labels(keywords, provider = "openai")
} # }
```
