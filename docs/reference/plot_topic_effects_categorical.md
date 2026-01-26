# Plot Topic Effects for Categorical Variables

Creates a faceted plot showing how categorical variables affect topic
proportions.

## Usage

``` r
plot_topic_effects_categorical(
  effects_data,
  ncol = 2,
  height = 800,
  width = 1000,
  title = "Category Effects",
  base_font_size = 14
)
```

## Arguments

- effects_data:

  Data frame with columns: topic, value, proportion, lower, upper

- ncol:

  Number of columns for faceting (default: 2)

- height:

  Plot height in pixels (default: 800)

- width:

  Plot width in pixels (default: 1000)

- title:

  Plot title (default: "Category Effects")

- base_font_size:

  Base font size in pixels for the plot theme (default: 14). Axis text
  and strip text will be base_font_size + 2.

## Value

A plotly object

## See also

Other topic_modeling:
[`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md),
[`plot_topic_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_probability.md),
[`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md)
