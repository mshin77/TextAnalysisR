# Plot TF-IDF Keywords

Creates a horizontal bar plot of top keywords by TF-IDF score.

## Usage

``` r
plot_tfidf_keywords(tfidf_data, title = NULL, normalized = FALSE)
```

## Arguments

- tfidf_data:

  Data frame from extract_keywords_tfidf()

- title:

  Plot title (default: "Top Keywords by TF-IDF Score")

- normalized:

  Logical, whether scores are normalized (for label) (default: FALSE)

## Value

A plotly bar chart
