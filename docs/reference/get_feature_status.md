# Get Feature Status

Returns availability status for all optional features.

## Usage

``` r
get_feature_status()
```

## Value

Named list with feature availability

## Examples

``` r
if (interactive()) {
  status <- get_feature_status()
  print(status)
}
```
