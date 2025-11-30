# Plot Word Frequency

Creates a bar plot showing the most frequent words in a document-feature
matrix (dfm).

## Usage

``` r
plot_word_frequency(dfm_object, n = 20, height = NULL, width = NULL, ...)
```

## Arguments

- dfm_object:

  A document-feature matrix created by quanteda::dfm().

- n:

  The number of top words to display (default: 20).

- height:

  The height of the resulting Plotly plot, in pixels (default: 800).

- width:

  The width of the resulting Plotly plot, in pixels (default: 1000).

- ...:

  Additional arguments passed to plotly::ggplotly().

## Value

A plotly object showing word frequency.

## Examples

``` r
if (interactive()) {
  texts <- c("mathematics technology", "education technology", "learning support")
  dfm <- quanteda::dfm(quanteda::tokens(texts))
  plot <- plot_word_frequency(dfm, n = 5)
  print(plot)
}
```
