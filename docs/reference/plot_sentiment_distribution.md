# Plot Sentiment Distribution

Creates a bar plot showing the distribution of sentiment
classifications.

## Usage

``` r
plot_sentiment_distribution(sentiment_data, title = "Sentiment Distribution")
```

## Arguments

- sentiment_data:

  Data frame from analyze_sentiment() or with 'sentiment' column

- title:

  Plot title (default: "Sentiment Distribution")

## Value

A ggplot2 bar chart

## Examples

``` r
# \donttest{
abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
sentiment_data <- analyze_sentiment(abstracts)
sentiment_plot <- plot_sentiment_distribution(sentiment_data)
print(sentiment_plot)

# }
```
