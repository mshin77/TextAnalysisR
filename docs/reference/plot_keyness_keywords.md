# Plot Statistical Keyness

Creates a horizontal bar plot of distinctive keywords by keyness score.

## Usage

``` r
plot_keyness_keywords(keyness_data, title = NULL, group_label = NULL)
```

## Arguments

- keyness_data:

  Data frame from extract_keywords_keyness()

- title:

  Plot title (default: "Top Keywords by Keyness (G-squared)")

- group_label:

  Optional label for the target group (default: NULL)

## Value

A plotly bar chart
