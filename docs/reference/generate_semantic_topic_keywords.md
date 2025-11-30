# Generate Semantic Topic Keywords (c-TF-IDF)

Generate keywords for topics using c-TF-IDF (class-based TF-IDF),
similar to BERTopic. This method treats all documents in a topic as a
single document and calculates TF-IDF scores relative to other topics.

## Usage

``` r
generate_semantic_topic_keywords(
  texts,
  topic_assignments,
  n_keywords = 10,
  method = "c-tfidf"
)
```

## Arguments

- texts:

  A character vector of texts.

- topic_assignments:

  A vector of topic assignments.

- n_keywords:

  The number of keywords to extract per topic (default: 10).

- method:

  The representation method: "c-tfidf" (default), "tfidf", "mmr", or
  "frequency".

## Value

A list of keywords for each topic.
