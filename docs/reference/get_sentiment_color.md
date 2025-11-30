# Generate Sentiment Color Gradient

Generates a color based on sentiment score using a gradient from red
(negative) through gray (neutral) to green (positive).

## Usage

``` r
get_sentiment_color(score)
```

## Arguments

- score:

  Numeric sentiment score (typically -1 to 1)

## Value

Hex color string

## Examples

``` r
get_sentiment_color(-0.8)  # Red
#> [1] "#A35A44"
get_sentiment_color(0)     # Gray
#> [1] "#4BB543"
get_sentiment_color(0.8)   # Green
#> [1] "#1CB875"
```
