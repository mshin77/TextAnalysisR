# List Available Ollama Models

Lists all models currently installed in Ollama.

## Usage

``` r
list_ollama_models(verbose = FALSE)
```

## Arguments

- verbose:

  Logical, if TRUE, prints status messages.

## Value

Character vector of model names, or NULL if Ollama is unavailable.

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
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md),
[`validate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/validate_topic_labels_langgraph.md)

## Examples

``` r
if (FALSE) { # \dontrun{
models <- list_ollama_models()
print(models)
} # }
```
