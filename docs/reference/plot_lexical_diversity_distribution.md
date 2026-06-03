# Plot Lexical Diversity Distribution

Creates a boxplot showing the distribution of a lexical diversity
metric.

## Usage

``` r
plot_lexical_diversity_distribution(lexdiv_data, metric, title = NULL)
```

## Arguments

- lexdiv_data:

  Data frame from lexical_diversity_analysis()

- metric:

  Metric to plot. Recommended: "MTLD" or "MATTR" (text-length
  independent)

- title:

  Plot title (default: auto-generated)

## Value

A plotly boxplot

## Examples

``` r
# \donttest{
abstracts <- TextAnalysisR::SpecialEduTech$abstract[1:10]
tokens <- quanteda::tokens(quanteda::corpus(abstracts))
diversity_result <- lexical_diversity_analysis(tokens, texts = abstracts)
diversity_plot <- plot_lexical_diversity_distribution(
  diversity_result$lexical_diversity, "MTLD"
)
print(diversity_plot)

# }
```
