# Create Message Data Table

Creates a formatted DT::datatable displaying an informational message.
Useful for showing status messages in place of empty tables.

## Usage

``` r
create_message_table(message, font_size = "16px", color = "#6c757d")
```

## Arguments

- message:

  Character string message to display

- font_size:

  Font size (default: "16px")

- color:

  Text color (default: "#6c757d")

## Value

A DT::datatable object

## Examples

``` r
if (FALSE) { # \dontrun{
create_message_table("No data available. Please run analysis first.")
} # }
```
