# Generate Survey Items from Topic Terms

Convenience wrapper for
[`generate_topic_content`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md)
with `content_type = "survey_item"`. Generates Likert-scale survey items
for scale development.

## Usage

``` r
generate_survey_items(
  topic_terms_df,
  topic_var = "topic",
  term_var = "term",
  weight_var = "beta",
  provider = c("openai", "ollama"),
  model = "gpt-3.5-turbo",
  temperature = 0,
  system_prompt = NULL,
  user_prompt_template = NULL,
  max_tokens = 150,
  api_key = NULL,
  verbose = TRUE
)
```

## Arguments

- topic_terms_df:

  A data frame with topic terms, containing columns for topic
  identifier, term, and optionally term weight (beta).

- topic_var:

  Name of the column containing topic identifiers (default: "topic").

- term_var:

  Name of the column containing terms (default: "term").

- weight_var:

  Name of the column containing term weights (default: "beta").

- provider:

  LLM provider: "openai" or "ollama" (default: "openai").

- model:

  Model name. For OpenAI: "gpt-3.5-turbo", "gpt-4", etc. For Ollama:
  "llama3", "mistral", etc.

- temperature:

  Sampling temperature (0-2). Lower = more deterministic (default: 0).

- system_prompt:

  Custom system prompt. If NULL, uses default for content_type.

- user_prompt_template:

  Custom user prompt template with {terms} placeholder. If NULL, uses
  default for content_type.

- max_tokens:

  Maximum tokens for response (default: 150).

- api_key:

  OpenAI API key. If NULL, reads from OPENAI_API_KEY environment
  variable.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A data frame with generated survey items joined to original topic terms.

## See also

[`generate_topic_content`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_content.md)

Other ai:
[`analyze_contrastive_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/analyze_contrastive_similarity.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`create_label_selection_data()`](https://mshin77.github.io/TextAnalysisR/reference/create_label_selection_data.md),
[`format_label_candidates()`](https://mshin77.github.io/TextAnalysisR/reference/format_label_candidates.md),
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
survey_items <- generate_survey_items(
  topic_terms_df = top_terms,
  provider = "openai",
  model = "gpt-3.5-turbo"
)
} # }
```
