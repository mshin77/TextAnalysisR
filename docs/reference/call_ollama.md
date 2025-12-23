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

## See also

Other ai:
[`analyze_contrastive_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_contrastive_similarity.md),
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
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md),
[`validate_topic_labels_langgraph()`](https://mshin77.github.io/TextAnalysisR/reference/validate_topic_labels_langgraph.md)

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
