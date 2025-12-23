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

## See also

Other ai:
[`analyze_contrastive_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_contrastive_similarity.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`create_label_selection_data()`](https://mshin77.github.io/TextAnalysisR/reference/create_label_selection_data.md),
[`format_label_candidates()`](https://mshin77.github.io/TextAnalysisR/reference/format_label_candidates.md),
[`generate_survey_items()`](https://mshin77.github.io/TextAnalysisR/reference/generate_survey_items.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`generate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels_langgraph.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md)

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
