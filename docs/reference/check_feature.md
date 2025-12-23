# Check Feature Status

Checks if a specific optional feature is available in the current
environment.

## Usage

``` r
check_feature(feature)
```

## Arguments

- feature:

  Character: "python", "ollama", "langgraph", "pdf_tables", "embeddings"

## Value

Logical TRUE if feature is available

## Examples

``` r
if (check_feature("ollama")) {
  # Use AI-powered labeling
}
#> NULL
```
