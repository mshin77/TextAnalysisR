# Calculate Topic Probabilities

Extracts and summarizes topic probabilities (gamma values) from an STM
model, returning a formatted data table of mean topic prevalence.

## Usage

``` r
calculate_topic_probability(stm_model, top_n = 10, verbose = TRUE, ...)
```

## Arguments

- stm_model:

  A fitted STM model object from stm::stm().

- top_n:

  Number of top topics to display by prevalence (default: 10).

- verbose:

  Logical, if TRUE prints progress messages (default: TRUE).

- ...:

  Additional arguments passed to tidytext::tidy().

## Value

A DT::datatable showing topics and their mean gamma (prevalence) values,
rounded to 3 decimal places.

## Examples

``` r
if (interactive()) {
  data <- TextAnalysisR::SpecialEduTech
  united <- unite_cols(data, c("title", "keyword", "abstract"))
  tokens <- prep_texts(united, text_field = "united_texts")
  dfm_obj <- quanteda::dfm(tokens)
  stm_data <- quanteda::convert(dfm_obj, to = "stm")

  topic_model <- stm::stm(
    documents = stm_data$documents,
    vocab = stm_data$vocab,
    K = 10,
    verbose = FALSE
  )

  prob_table <- calculate_topic_probability(topic_model, top_n = 10)
  print(prob_table)
}
```
