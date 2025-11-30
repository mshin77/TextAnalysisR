# Create Formatted Analysis Data Table

Creates a consistently formatted DT::datatable for analysis results with
export buttons and optional numeric formatting.

## Usage

``` r
create_analysis_datatable(
  data,
  colnames = NULL,
  numeric_cols = NULL,
  digits = 3,
  export_formats = c("copy", "csv", "excel", "pdf", "print"),
  page_length = 25,
  font_size = "16px"
)
```

## Arguments

- data:

  Data frame to display

- colnames:

  Optional character vector of column names for display

- numeric_cols:

  Optional character vector of numeric columns to round

- digits:

  Number of digits for rounding numeric columns (default: 3)

- export_formats:

  Character vector of export formats (default: c('copy', 'csv', 'excel',
  'pdf', 'print'))

- page_length:

  Number of rows per page (default: 25)

- font_size:

  Font size for table cells (default: "16px")

## Value

A DT::datatable object

## Examples

``` r
if (FALSE) { # \dontrun{
df <- data.frame(term = c("word1", "word2"), score = c(0.123456, 0.789012))
create_analysis_datatable(df, numeric_cols = "score", digits = 3)
} # }
```
