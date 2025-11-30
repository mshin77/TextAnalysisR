# Validate Column Name

Validates column names to prevent code injection through formula
construction. Ensures column names follow R naming conventions and
contain no malicious patterns.

## Usage

``` r
validate_column_name(col_name)
```

## Arguments

- col_name:

  Character string containing the column name

## Value

TRUE if valid, stops with error if invalid

## Security

Protects against formula injection attacks where malicious column names
could execute arbitrary code when used in model formulas. Part of NIST
SI-10 input validation.

## Examples

``` r
if (FALSE) { # \dontrun{
validate_column_name("age")
validate_column_name("my_variable")
} # }
```
