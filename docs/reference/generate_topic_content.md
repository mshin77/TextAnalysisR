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
  provider = c("openai", "gemini"),
  model = "gpt-4.1-mini",
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

  LLM provider: "openai" or "gemini" (default: "openai").

- model:

  Model name. For OpenAI: "gpt-4.1-mini", "gpt-4", etc. For Gemini:
  "gemini-2.5-flash-lite", "gemini-2.5-flash", etc.

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

  API key for the selected provider. If NULL, reads from OPENAI_API_KEY
  or GEMINI_API_KEY environment variable.

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

Requires an API key set via the `api_key` parameter or the relevant
environment variable (OPENAI_API_KEY or GEMINI_API_KEY).

## See also

[`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md)
for the step that creates topic labels;
[`get_content_type_prompt()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_prompt.md)
and
[`get_content_type_user_template()`](https://mshin77.github.io/TextAnalysisR/reference/get_content_type_user_template.md)
to inspect or override default prompts

## Examples

``` r
if (interactive()) {
# Generate survey items
survey_items <- generate_topic_content(
  topic_terms_df = top_terms,
  content_type = "survey_item",
  provider = "openai",
  model = "gpt-4.1-mini"
)

# Generate research questions
research_qs <- generate_topic_content(
  topic_terms_df = top_terms,
  content_type = "research_question",
  provider = "gemini",
  model = "gemini-2.5-flash-lite"
)

# Generate with custom prompt
custom_content <- generate_topic_content(
  topic_terms_df = top_terms,
  content_type = "custom",
  system_prompt = "You are an expert in educational policy...",
  user_prompt_template = "Based on {terms}, generate a learning objective:"
)
}
```
