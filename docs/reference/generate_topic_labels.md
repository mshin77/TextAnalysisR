# Generate Topic Labels Using OpenAI's API

This function generates descriptive labels for each topic based on their
top terms using OpenAI's ChatCompletion API.

## Usage

``` r
generate_topic_labels(
  top_topic_terms,
  model = "gpt-3.5-turbo",
  system = NULL,
  user = NULL,
  temperature = 0.5,
  openai_api_key = NULL,
  verbose = TRUE
)
```

## Arguments

- top_topic_terms:

  A data frame containing the top terms for each topic.

- model:

  A character string specifying which OpenAI model to use (default:
  "gpt-3.5-turbo").

- system:

  A character string containing the system prompt for the OpenAI API. If
  NULL, the function uses the default system prompt.

- user:

  A character string containing the user prompt for the OpenAI API. If
  NULL, the function uses the default user prompt.

- temperature:

  A numeric value controlling the randomness of the output (default:
  0.5).

- openai_api_key:

  A character string containing the OpenAI API key. If NULL, the
  function attempts to load the key from the OPENAI_API_KEY environment
  variable or the .env file in the working directory.

- verbose:

  Logical, if TRUE, prints progress messages.

## Value

A data frame containing the top terms for each topic along with their
generated labels.

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech

  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )

  tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")

  dfm_object <- quanteda::dfm(tokens)

  out <- quanteda::convert(dfm_object, to = "stm")

stm_15 <- stm::stm(
  data = out$meta,
  documents = out$documents,
  vocab = out$vocab,
  max.em.its = 75,
  init.type = "Spectral",
  K = 15,
  prevalence = ~ reference_type + s(year),
  verbose = TRUE)

top_topic_terms <- TextAnalysisR::get_topic_terms(
  stm_model = stm_15,
  top_term_n = 10,
  verbose = TRUE
  )

top_labeled_topic_terms <- TextAnalysisR::generate_topic_labels(
  top_topic_terms,
  model = "gpt-3.5-turbo",
  temperature = 0.5,
  openai_api_key = "your_openai_api_key",
  verbose = TRUE)
print(top_labeled_topic_terms)

top_labeled_topic_terms <- TextAnalysisR::generate_topic_labels(
  top_topic_terms,
  model = "gpt-3.5-turbo",
  temperature = 0.5,
  verbose = TRUE)
print(top_labeled_topic_terms)
}
```
