# Generate Topic Keywords

Internal function to generate keywords for topics using TF-IDF analysis.

## Usage

``` r
generate_topic_keywords(texts, topic_assignments, n_keywords = 10)
```

## Arguments

- texts:

  A character vector of texts.

- topic_assignments:

  A vector of topic assignments.

- n_keywords:

  The number of keywords to extract per topic.

## Value

A list of keywords for each topic.
