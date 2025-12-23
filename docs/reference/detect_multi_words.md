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

## See also

Other preprocessing:
[`get_available_dfm()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_dfm.md),
[`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md),
[`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md),
[`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md),
[`unite_cols()`](https://mshin77.github.io/TextAnalysisR/reference/unite_cols.md)

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
