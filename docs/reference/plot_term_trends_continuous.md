# Plot Term Frequency Trends by Continuous Variable

Creates a faceted line plot showing how term frequencies vary across a
continuous variable (e.g., year, time period).

## Usage

``` r
plot_term_trends_continuous(
  term_data,
  continuous_var,
  terms = NULL,
  title = NULL,
  height = 600,
  width = NULL
)
```

## Arguments

- term_data:

  Data frame containing term frequencies with columns: continuous_var,
  term, and word_frequency

- continuous_var:

  Name of the continuous variable column

- terms:

  Character vector of terms to display (optional, filters if provided)

- title:

  Plot title (default: NULL, auto-generated)

- height:

  Plot height in pixels (default: 600)

- width:

  Plot width in pixels (default: NULL, auto)

## Value

A plotly object with faceted line plots

## Examples

``` r
# \donttest{
term_df <- data.frame(
  year = rep(2010:2020, each = 3),
  term = rep(c("learning", "education", "technology"), 11),
  word_frequency = sample(10:100, 33, replace = TRUE)
)
plot_term_trends_continuous(term_df, "year", c("learning", "education"))

# }
```
