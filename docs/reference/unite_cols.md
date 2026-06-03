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
# \donttest{
  mydata <- TextAnalysisR::SpecialEduTech

  united_tbl <- TextAnalysisR::unite_cols(
    mydata,
    listed_vars = c("title", "keyword", "abstract")
  )
  print(united_tbl)
#> # A tibble: 490 × 7
#>    united_texts               title keyword abstract reference_type author  year
#>    <chr>                      <chr> <chr>   <chr>    <chr>          <chr>  <dbl>
#>  1 Dyscalculia and the minic… Dysc… "Arith… Notes t… journal_artic… Block…  1980
#>  2 The effects of computer-a… The … "locus… This st… thesis         Bukat…  1981
#>  3 Computer Assisted Instruc… Comp… "Compu… Results… journal_artic… Watki…  1981
#>  4 Arc-Ed Curriculum: Applic… Arc-… "Compu… The Arc… journal_artic… Chaff…  1982
#>  5 ARC-ED curriculum: the ap… ARC-… "Elect… This ar… journal_artic… Chaff…  1982
#>  6 The Effect of the Hand-he… The … ""      The pur… thesis         Golde…  1982
#>  7 A review of some traditio… A re… "tradi… Discuss… journal_artic… Neal,…  1982
#>  8 A study of the effectiven… A st… "micro… The pur… thesis         Engle…  1983
#>  9 The influence of computer… The … "compu… The eff… thesis         Foste…  1983
#> 10 Using Computer Software t… Usin… "Compu… The art… journal_artic… Pomme…  1983
#> # ℹ 480 more rows
# }
```
