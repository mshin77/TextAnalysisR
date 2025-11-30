# Validate User-Selected Topic Labels

Uses LangGraph workflow to validate user-selected topic labels using
LLM.

## Usage

``` r
validate_topic_labels_langgraph(
  user_labels,
  topic_terms,
  ollama_model = "llama3",
  ollama_base_url = "http://localhost:11434",
  envname = "textanalysisr-env"
)
```

## Arguments

- user_labels:

  Character vector of user-selected labels for each topic

- topic_terms:

  List of character vectors with top terms for each topic

- ollama_model:

  Character string, Ollama model name (default: "llama3")

- ollama_base_url:

  Character string, Ollama API URL (default: "http://localhost:11434")

- envname:

  Character string, Python virtual environment name (default:
  "langgraph-env")

## Value

List with:

- success: Logical, TRUE if validation completed

- validation_metrics: List with coherence and distinctiveness scores

- error: Error message (if failed)

## Details

Validation metrics include:

- coherence_scores: How well labels match term distributions (0-10
  scale)

- distinctiveness_scores: How unique/specific labels are (0-10 scale)

- overall_quality: Average of coherence and distinctiveness

## Examples

``` r
if (FALSE) { # \dontrun{
user_labels <- c("Education Policy", "Healthcare Services", "Climate Action")
topic_terms <- list(
  c("education", "student", "learning"),
  c("health", "medical", "patient"),
  c("environment", "climate", "carbon")
)

validation <- validate_topic_labels_langgraph(
  user_labels = user_labels,
  topic_terms = topic_terms
)

print(validation$validation_metrics$overall_quality)
} # }
```
