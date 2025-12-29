# Check Feature Status

Checks if a specific optional feature is available in the current
environment.

## Usage

``` r
check_feature(feature)
```

## Arguments

- feature:

  Character: "python", "ollama", "pdf_tables", "embeddings",
  "sentiment_deep"

## Value

Logical TRUE if feature is available

## Examples

``` r
if (check_feature("ollama")) {
  # Use AI-powered labeling
}
#> NULL
```
