# Plot Part-of-Speech Tag Frequencies

Creates a bar plot showing the frequency distribution of part-of-speech
tags.

## Usage

``` r
plot_pos_frequencies(
  pos_data,
  top_n = 20,
  title = "Part-of-Speech Tag Frequency",
  color = "#337ab7",
  height = 500,
  width = NULL
)
```

## Arguments

- pos_data:

  Data frame containing POS data with columns:

  - `pos`: Part-of-speech tag

  - `n`: (optional) Pre-computed frequency count

  If `n` is not present, frequencies will be computed from the data.

- top_n:

  Number of top POS tags to display (default: 20)

- title:

  Plot title (default: "Part-of-Speech Tag Frequency")

- color:

  Bar color (default: "#337ab7")

- height:

  Plot height in pixels (default: 500)

- width:

  Plot width in pixels (default: NULL for auto)

## Value

A plotly object

## Examples

``` r
if (interactive()) {
  pos_df <- data.frame(
    pos = c("NOUN", "VERB", "ADJ", "ADV", "PRON"),
    n = c(500, 400, 250, 150, 100)
  )
  plot_pos_frequencies(pos_df)
}
```
