# Build a topic-term data frame from any supported topic model

Unified helper that produces the long-format
`data.frame(topic, term, beta)` expected by
[`generate_topic_labels()`](https://mshin77.github.io/TextAnalysisR/reference/generate_topic_labels.md)
from an STM model or an embedding result. Dispatches on the object's
structure:

- STM model (has `$beta$logbeta` and `$vocab`) -\> top terms via
  [`stm::labelTopics()`](https://rdrr.io/pkg/stm/man/labelTopics.html)
  FREX

- Embedding result (has `$topic_keywords`) -\> c-TF-IDF keywords with
  rank-derived pseudo-beta

## Usage

``` r
extract_topic_terms_df(model, n = 7)
```

## Arguments

- model:

  A topic model object (STM fit or embedding result).

- n:

  Number of top terms per topic (default 7).

## Value

`data.frame(topic, term, beta)` in long format.
