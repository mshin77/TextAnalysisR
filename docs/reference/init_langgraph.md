# Initialize LangGraph for Current Session

Initializes LangGraph/LangChain/Ollama modules for current R session.
Use only for LangGraph workflows. PDF/embeddings load automatically.

## Usage

``` r
init_langgraph(envname = "textanalysisr-env")
```

## Arguments

- envname:

  Character string name of the virtual environment (default:
  "textanalysisr-env")

## Value

Invisible list with LangGraph/LangChain/Ollama modules

## Examples

``` r
if (FALSE) { # \dontrun{
lg <- init_langgraph()
} # }
```
