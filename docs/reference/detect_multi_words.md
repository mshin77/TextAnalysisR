# Detect Multi-Word Expressions

This function detects multi-word expressions (collocations) of specified
sizes that appear at least a specified number of times in the provided
tokens.

## Usage

``` r
detect_multi_words(tokens, size = 2:5, min_count = 2)
```

## Arguments

- tokens:

  A `tokens` object from the `quanteda` package.

- size:

  A numeric vector specifying the sizes of the collocations to detect
  (default: 2:5).

- min_count:

  The minimum number of occurrences for a collocation to be considered
  (default: 2).

## Value

A character vector of detected collocations.

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech

  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )

  tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")

  collocations <- TextAnalysisR::detect_multi_words(tokens, size = 2:5, min_count = 2)
  print(collocations)
}
```
