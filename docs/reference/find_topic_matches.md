# Find Similar Topics

This function finds the most similar topics to a given query using
semantic similarity analysis. It works with both semantic topic models
and traditional STM models by creating topic representations using
transformer embeddings and calculating cosine similarity scores.

## Usage

``` r
find_topic_matches(
  topic_model,
  query,
  top_n = 10,
  method = "cosine",
  embedding_model = "all-MiniLM-L6-v2",
  include_terms = TRUE
)
```

## Arguments

- topic_model:

  A topic model object (semantic topic model or STM model).

- query:

  A character string representing the query topic.

- top_n:

  The number of similar topics to return (default: 10).

- method:

  The similarity method: "cosine", "euclidean", "embedding".

- embedding_model:

  The embedding model to use for query encoding (default:
  "all-MiniLM-L6-v2").

- include_terms:

  Logical, whether to include topic terms in the similarity calculation
  (default: TRUE).

## Value

A list containing similar topics and their similarity scores.

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech
  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )
  texts <- united_tbl$united_texts

  topic_model <- TextAnalysisR::fit_embedding_topics(
    texts = texts,
    method = "semantic_style",
    n_topics = 8
  )

  similar_topics <- TextAnalysisR::find_similar_topics(
    topic_model = topic_model,
    query = "mathematical learning",
    top_n = 5
  )

  print(similar_topics)
}
```
