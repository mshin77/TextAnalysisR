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
# \donttest{
df <- data.frame(term = c("word1", "word2"), score = c(0.123456, 0.789012))
create_analysis_datatable(df, numeric_cols = "score", digits = 3)

{"x":{"filter":"none","vertical":false,"extensions":["Buttons"],"data":[["word1","word2"],[0.123456,0.789012]],"container":"<table class=\"display\">\n  <thead>\n    <tr><\/tr>\n  <\/thead>\n<\/table>","options":{"scrollX":true,"pageLength":25,"dom":"Bfrtip","buttons":["copy","csv","excel","pdf","print"],"columnDefs":[{"targets":1,"render":"function(data, type, row, meta) {\n    return type !== 'display' ? data : DTWidget.formatRound(data, 3, 3, \",\", \".\", null);\n  }"},{"className":"dt-right","targets":1},{"name":"term","targets":0},{"name":"score","targets":1}],"order":[],"autoWidth":false,"orderClasses":false,"rowCallback":"function(row, data, displayNum, displayIndex, dataIndex) {\nvar value=data[0]; $(this.api().cell(row, 0).node()).css({'font-size':'16px'});\nvar value=data[1]; $(this.api().cell(row, 1).node()).css({'font-size':'16px'});\n}"},"selection":{"mode":"multiple","selected":null,"target":"row","selectable":null}},"evals":["options.columnDefs.0.render","options.rowCallback"],"jsHooks":[]}# }
```
