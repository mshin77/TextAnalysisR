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
status <- get_feature_status()
#> Downloading uv...
#> Done!
print(status)
#> $python
#> [1] FALSE
#> 
#> $ollama
#> [1] FALSE
#> 
#> $langgraph
#> [1] FALSE
#> 
#> $pdf_tables
#> [1] FALSE
#> 
#> $embeddings
#> [1] FALSE
#> 
#> $sentiment_deep
#> [1] FALSE
#> 
#> $web
#> [1] FALSE
#> 
#> $local
#> [1] TRUE
#> 
```
