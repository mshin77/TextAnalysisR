# Select Top Terms for Each Topic

This function selects the top terms for each topic based on their word
probability distribution (beta).

## Usage

``` r
get_topic_terms(stm_model, top_term_n = 10, verbose = TRUE, ...)
```

## Arguments

- stm_model:

  An STM model object.

- top_term_n:

  The number of top terms to display for each topic (default: 10).

- verbose:

  Logical, if TRUE, prints progress messages.

- ...:

  Further arguments passed to tidytext::tidy.

## Value

A data frame containing the top terms for each topic.

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

  stm_15 <- TextAnalysisR::create_stm_model(
  dfm_object,
  topic_n = 15,
  max.em.its = 75,
  categorical_var = "reference_type",
  continuous_var = "year",
  verbose = TRUE
  )

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
print(top_topic_terms)
}
```
