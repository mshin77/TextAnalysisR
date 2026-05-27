# Get Topic Prevalence (Gamma) from STM Model

Extracts topic prevalence values (gamma/theta) from a fitted STM model,
returning mean prevalence for each topic as a data frame.

## Usage

``` r
get_topic_prevalence(stm_model, category = NULL, include_theta = FALSE)
```

## Arguments

- stm_model:

  A fitted STM model object from stm::stm().

- category:

  Optional character string to add as a category column.

- include_theta:

  Logical, if TRUE includes document-topic matrix (default: FALSE).

## Value

A data frame with columns:

- topic:

  Topic number

- gamma:

  Mean topic prevalence across documents

- category:

  Category label (if provided)

## Examples

``` r
if (interactive() && requireNamespace("stm", quietly = TRUE)) {
  # Requires fitting an STM model first; uses 'stm::gadarian' for demo
  data("gadarian", package = "stm")
  proc <- stm::textProcessor(gadarian$open.ended.response, metadata = gadarian)
  prep <- stm::prepDocuments(proc$documents, proc$vocab, proc$meta)
  topic_model <- stm::stm(prep$documents, prep$vocab, K = 3,
                           data = prep$meta, max.em.its = 5,
                           verbose = FALSE)
  prevalence <- get_topic_prevalence(topic_model)
  prevalence_label <- get_topic_prevalence(topic_model, category = "demo")
}
```
