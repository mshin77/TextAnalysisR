# Generate Content from Topic Terms

Uses Large Language Models (LLMs) to generate various types of content
based on topic model terms. Supports multiple content types with
optimized default prompts, or fully custom prompts.

## Usage

``` r
generate_topic_content(
  topic_terms_df,
  content_type = c("survey_item", "research_question", "theme_description",
    "policy_recommendation", "interview_question", "custom"),
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
  output_var = NULL,
  verbose = TRUE
)
```

## Arguments

- topic_terms_df:

  A data frame with topic terms, containing columns for topic
  identifier, term, and optionally term weight (beta).

- content_type:

  Type of content to generate. One of:

  "survey_item"

  :   Likert-scale survey items for scale development

  "research_question"

  :   Research questions for literature review

  "theme_description"

  :   Theme descriptions for qualitative analysis

  "policy_recommendation"

  :   Policy recommendations for policy analysis

  "interview_question"

  :   Interview questions for qualitative research

  "custom"

  :   Custom content using user-provided prompts

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

- output_var:

  Name of the output column (default: based on content_type).

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A data frame with generated content joined to original topic terms.

## Details

The function generates one piece of content per unique topic. Each
content type has optimized default prompts, but these can be overridden
with custom prompts.

For OpenAI, requires an API key set via the `api_key` parameter or
OPENAI_API_KEY environment variable (can be loaded from .env file).

For Ollama, requires a local Ollama installation with the specified
model.

## See also

Other ai:
[`call_gemini_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_gemini_chat.md),
[`call_llm_api()`](https://mshin77.github.io/TextAnalysisR/reference/call_llm_api.md),
[`call_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/call_ollama.md),
[`call_openai_chat()`](https://mshin77.github.io/TextAnalysisR/reference/call_openai_chat.md),
[`check_ollama()`](https://mshin77.github.io/TextAnalysisR/reference/check_ollama.md),
[`get_api_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_api_embeddings.md),
[`get_best_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_best_embeddings.md),
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md),
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md),
[`get_recommended_ollama_model()`](https://mshin77.github.io/TextAnalysisR/reference/get_recommended_ollama_model.md),
[`list_ollama_models()`](https://mshin77.github.io/TextAnalysisR/reference/list_ollama_models.md),
[`run_rag_search()`](https://mshin77.github.io/TextAnalysisR/reference/run_rag_search.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Generate survey items
survey_items <- generate_topic_content(
  topic_terms_df = top_terms,
  content_type = "survey_item",
  provider = "openai",
  model = "gpt-3.5-turbo"
)

# Generate research questions
research_qs <- generate_topic_content(
  topic_terms_df = top_terms,
  content_type = "research_question",
  provider = "ollama",
  model = "llama3"
)

# Generate with custom prompt
custom_content <- generate_topic_content(
  topic_terms_df = top_terms,
  content_type = "custom",
  system_prompt = "You are an expert in educational policy...",
  user_prompt_template = "Based on {terms}, generate a learning objective:"
)
} # }
```
