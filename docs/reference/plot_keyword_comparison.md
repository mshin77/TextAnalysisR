# Plot Keyword Comparison (TF-IDF vs Frequency)

Creates a grouped bar plot comparing TF-IDF scores with term
frequencies.

## Usage

``` r
plot_keyword_comparison(
  tfidf_data,
  top_n = 10,
  title = NULL,
  normalized = FALSE
)
```

## Arguments

- tfidf_data:

  Data frame from extract_keywords_tfidf()

- top_n:

  Number of keywords to display (default: 10)

- title:

  Plot title (default: auto-generated)

- normalized:

  Logical, whether TF-IDF scores are normalized (default: FALSE)

## Value

A plotly grouped bar chart
