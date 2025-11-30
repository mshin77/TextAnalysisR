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
if (FALSE) { # \dontrun{
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:10]
corp <- quanteda::corpus(texts)
toks <- quanteda::tokens(corp)
dfm_obj <- quanteda::dfm(toks)
result <- lexical_diversity_analysis(dfm_obj)
plot <- plot_lexical_diversity_distribution(result$lexical_diversity, "MTLD")
print(plot)
} # }
```
