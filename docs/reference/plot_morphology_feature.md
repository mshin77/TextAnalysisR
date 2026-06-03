# Plot Morphology Feature Distribution

Creates a bar chart showing the distribution of a morphological feature
using consistent package styling.

## Usage

``` r
plot_morphology_feature(data, feature, title = NULL, colors = NULL)
```

## Arguments

- data:

  Data frame with morph\_\* columns from extract_morphology().

- feature:

  Character; feature name (e.g., "Number", "Tense").

- title:

  Character; plot title (auto-generated if NULL).

- colors:

  Named character vector of custom colors for feature values.

## Value

A plotly object.

## Examples

``` r
if (interactive()) {
  tokens <- quanteda::tokens(TextAnalysisR::SpecialEduTech$abstract[1])
  morphology_data <- extract_morphology(tokens)
  plot_morphology_feature(morphology_data, "Tense")
}
```
