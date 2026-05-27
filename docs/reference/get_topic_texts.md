# Convert Topic Terms to Text Strings

Concatenates top terms for each topic into text strings suitable for
embedding generation. Useful for creating topic representations for
semantic similarity analysis.

## Usage

``` r
get_topic_texts(
  top_terms_df,
  topic_var = "topic",
  term_var = "term",
  weight_var = NULL,
  sep = " ",
  top_n = NULL
)
```

## Arguments

- top_terms_df:

  A data frame containing top terms for topics, typically output from
  [`get_topic_terms`](https://mshin77.github.io/TextAnalysisR/reference/get_topic_terms.md).

- topic_var:

  Name of the column containing topic identifiers (default: "topic").

- term_var:

  Name of the column containing terms (default: "term").

- weight_var:

  Optional name of column with term weights (e.g., "beta"). If provided,
  terms are ordered by weight before concatenation.

- sep:

  Separator between terms (default: " ").

- top_n:

  Optional number of top terms to include per topic (default: NULL, uses
  all).

## Value

A character vector of topic text strings, one per topic, ordered by
topic number.

## Examples

``` r
# Topic-term frame as produced by get_topic_terms()
top_terms <- data.frame(
  topic = c(1, 1, 1, 2, 2, 2),
  term  = c("calculator", "arithmetic", "elementary",
            "computer", "instruction", "multiplication"),
  prob  = c(0.10, 0.08, 0.07, 0.12, 0.09, 0.06)
)
get_topic_texts(top_terms)
#> [1] "calculator arithmetic elementary"    "computer instruction multiplication"
get_topic_texts(top_terms, weight_var = "prob", top_n = 2)
#> [1] "calculator arithmetic" "computer instruction" 
```
