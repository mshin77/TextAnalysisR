# Summarize Morphology Features

Creates a summary table of morphological feature distributions with
counts and percentages for each feature value.

## Usage

``` r
summarize_morphology(data, features = NULL)
```

## Arguments

- data:

  Data frame with morph\_\* columns from extract_morphology().

- features:

  Character vector of features to summarize. If NULL, all available
  morph\_\* columns are used.

## Value

A data frame with Feature, Value, Count, and Percentage columns.

## Examples

``` r
if (interactive()) {
  tokens <- quanteda::tokens(TextAnalysisR::SpecialEduTech$abstract[1])
  morphology_data <- extract_morphology(tokens)
  summary_table <- summarize_morphology(morphology_data)
  print(summary_table)
}
```
