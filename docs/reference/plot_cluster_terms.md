# Plot Cluster Top Terms

Creates a horizontal bar plot showing the top terms in a cluster or
document group.

## Usage

``` r
plot_cluster_terms(
  terms,
  cluster_id = NULL,
  title = NULL,
  n_terms = 10,
  color = "#337ab7",
  height = 500,
  width = NULL
)
```

## Arguments

- terms:

  Named numeric vector of term frequencies, or data frame with 'term'
  and 'frequency' columns

- cluster_id:

  Cluster identifier for the title (default: NULL)

- title:

  Custom title (default: NULL, auto-generated from cluster_id)

- n_terms:

  Number of top terms to display (default: 10)

- color:

  Bar color (default: "#337ab7")

- height:

  Plot height in pixels (default: 500)

- width:

  Plot width in pixels (default: NULL for auto)

## Value

A plotly object
