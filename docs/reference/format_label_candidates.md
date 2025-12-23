# Format Label Candidates for Display

Helper function to format LangGraph label candidates for display in
Shiny UI.

## Usage

``` r
format_label_candidates(label_candidates)
```

## Arguments

- label_candidates:

  List of label candidate objects from generate_topic_labels_langgraph()

## Value

Data frame with columns:

- topic_index: Integer, topic number

- top_terms: Character, comma-separated top terms

- label: Character, suggested label

- reasoning: Character, LLM explanation

- candidate_number: Integer, candidate rank (1-3)

## See also

Other ai:
[`analyze_contrastive_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_contrastive_similarity.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`create_label_selection_data()`](https://mshin77.github.io/TextAnalysisR/reference/create_label_selection_data.md),
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
df <- format_label_candidates(result$label_candidates)
print(df)
} # }
```
