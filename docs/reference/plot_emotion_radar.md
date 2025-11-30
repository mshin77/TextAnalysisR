# Plot Emotion Radar Chart

Creates a polar/radar chart for NRC emotion analysis with optional
grouping.

## Usage

``` r
plot_emotion_radar(
  emotion_data,
  group_var = NULL,
  normalize = FALSE,
  title = "Emotion Analysis"
)
```

## Arguments

- emotion_data:

  Data frame with emotion scores (columns: emotion, total_score)

- group_var:

  Optional grouping variable column name for overlaid radars (default:
  NULL)

- normalize:

  Logical, whether to normalize scores to 0-100 scale (default: FALSE)

- title:

  Plot title (default: "Emotion Analysis")

## Value

A plotly polar chart
