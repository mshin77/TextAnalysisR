# Plot Per-Document Per-Topic Probabilities

This function generates a bar plot showing the prevalence of each topic
across all documents.

## Usage

``` r
plot_topic_probability(
  stm_model = NULL,
  gamma_data = NULL,
  top_n = 10,
  height = 800,
  width = 1000,
  topic_labels = NULL,
  colors = NULL,
  verbose = TRUE,
  ...
)
```

## Arguments

- stm_model:

  A fitted STM model object. where `stm_model` is a fitted Structural
  Topic Model created using
  [`stm::stm()`](https://rdrr.io/pkg/stm/man/stm.html).

- gamma_data:

  Optional pre-computed gamma data frame (default: NULL). If provided,
  used instead of stm_model.

- top_n:

  The number of topics to display, ordered by their mean prevalence.

- height:

  The height of the resulting Plotly plot, in pixels (default: 800).

- width:

  The width of the resulting Plotly plot, in pixels (default: 1000).

- topic_labels:

  Optional topic labels (default: NULL).

- colors:

  Optional color palette for topics (default: NULL).

- verbose:

  Logical, if TRUE, prints progress messages.

- ...:

  Further arguments passed to
  [`tidytext::tidy`](https://generics.r-lib.org/reference/tidy.html).

## Value

A `ggplot` object showing a bar plot of topic prevalence. Topics are
ordered by their mean gamma value (average prevalence across documents).

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

topic_probability_plot <- TextAnalysisR::plot_topic_probability(
 stm_model = stm_15,
 top_n = 10,
 height = 800,
 width = 1000,
 verbose = TRUE)

print(topic_probability_plot)
}
```
