# Create Label Selection UI Data

Creates a structured list for rendering label selection UI in Shiny.

## Usage

``` r
create_label_selection_data(label_candidates)
```

## Arguments

- label_candidates:

  List from generate_topic_labels_langgraph()

## Value

List of topic objects, each with:

- topic_number: Integer

- top_terms: Character vector

- candidates: List of candidate objects

## See also

Other ai:
[`analyze_contrastive_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_contrastive_similarity.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`format_label_candidates()`](https://mshin77.github.io/TextAnalysisR/reference/format_label_candidates.md),
[`generate_survey_items()`](https://mshin77.github.io/TextAnalysisR/reference/generate_survey_items.md),
[`generate_topic_content()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md),
[`generate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels_langgraph.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md),
[`validate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/validate_topic_labels_langgraph.md)

## Examples

``` r
if (FALSE) { # \dontrun{
result <- generate_topic_labels_langgraph(...)
ui_data <- create_label_selection_data(result$label_candidates)
} # }
```
