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
# \donttest{
status <- get_feature_status()
print(status)
#> $python
#> [1] FALSE
#> 
#> $ollama
#> [1] TRUE
#> 
#> $pdf_tables
#> [1] FALSE
#> 
#> $embeddings
#> [1] TRUE
#> 
#> $sentiment_deep
#> [1] TRUE
#> 
#> $web
#> [1] FALSE
#> 
#> $local
#> [1] TRUE
#> 
# }
```
