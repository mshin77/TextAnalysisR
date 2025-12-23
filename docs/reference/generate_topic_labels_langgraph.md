# Generate Topic Labels with LLM Assistance

Uses LangGraph workflow to generate multiple label candidates for topics
using local LLM (Ollama). Provides human-in-the-loop review of
suggestions.

## Usage

``` r
generate_topic_labels_langgraph(
  topic_terms,
  num_topics,
  ollama_model = "llama3",
  ollama_base_url = "http://localhost:11434",
  envname = "textanalysisr-env"
)
```

## Arguments

- topic_terms:

  List of character vectors, where each vector contains the top terms
  for a topic (from STM or other topic model)

- num_topics:

  Integer, number of topics

- ollama_model:

  Character string, name of Ollama model to use (default: "llama3")

- ollama_base_url:

  Character string, base URL for Ollama API (default:
  "http://localhost:11434")

- envname:

  Character string, name of Python virtual environment (default:
  "langgraph-env")

## Value

List with:

- success: Logical, TRUE if workflow completed successfully

- label_candidates: List of label candidate objects for each topic

- validation_metrics: Validation metrics (if available)

- error: Error message (if failed)

## Details

This function:

1.  Initializes LangGraph Python environment

2.  Calls Python workflow to generate label candidates

3.  Returns structured results for display in Shiny UI

4.  Allows human review and selection of labels

The workflow uses a StateGraph with nodes for:

- Label generation (LLM)

- Validation (LLM)

- Conditional revision based on quality metrics

## See also

Other ai:
[`analyze_contrastive_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_contrastive_similarity.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`create_label_selection_data()`](https://mshin77.github.io/TextAnalysisR/reference/create_label_selection_data.md),
[`format_label_candidates()`](https://mshin77.github.io/TextAnalysisR/reference/format_label_candidates.md),
[`generate_survey_items()`](https://mshin77.github.io/TextAnalysisR/reference/generate_survey_items.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md),
[`validate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/validate_topic_labels_langgraph.md)

## Examples

``` r
if (FALSE) { # \dontrun{
topic_terms <- list(
  c("education", "student", "learning", "teacher", "school"),
  c("health", "medical", "patient", "doctor", "treatment"),
  c("environment", "climate", "carbon", "emissions", "energy")
)

result <- generate_topic_labels_langgraph(
  topic_terms = topic_terms,
  num_topics = 3,
  ollama_model = "llama3"
)

if (result$success) {
  print(result$label_candidates)
}
} # }
```
