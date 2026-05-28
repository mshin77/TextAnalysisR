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

  Plot height in pixels (default: 800). Kept for backward compatibility.

- width:

  Plot width in pixels (default: 1000). Kept for backward compatibility.

- ...:

  Additional arguments (kept for backward compatibility).

## Value

A ggplot object showing word frequency.

## Examples

``` r
# \donttest{
  data(SpecialEduTech, package = "TextAnalysisR")
  texts <- SpecialEduTech$abstract[1:10]
  dfm <- quanteda::dfm(quanteda::tokens(texts))
  plot <- plot_word_frequency(dfm, n = 10)
  print(plot)

# }
```
