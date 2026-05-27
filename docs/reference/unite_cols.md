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

[`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md)
to tokenize and clean the united text;
[`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md)
to load source data first

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
