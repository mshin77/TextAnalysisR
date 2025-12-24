# Unite Text Columns

This function unites specified text columns in a data frame into a
single column named "united_texts" while retaining the original columns.

## Usage

``` r
unite_cols(df, listed_vars)
```

## Arguments

- df:

  A data frame that contains text data.

- listed_vars:

  A character vector of column names to be united into "united_texts".

## Value

A data frame with a new column "united_texts" created by uniting the
specified variables.

## See also

Other preprocessing:
[`get_available_dfm()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_dfm.md),
[`get_available_tokens()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_tokens.md),
[`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md),
[`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md),
[`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md)

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech

  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )
  print(united_tbl)
}
```
