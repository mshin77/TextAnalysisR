# Plot Word Probabilities by Topic

Creates a faceted bar plot showing the top terms and their probabilities
(beta values) for each topic in a topic model.

## Usage

``` r
plot_word_probability(
  top_topic_terms,
  topic_label = NULL,
  ncol = 3,
  height = 1200,
  width = 800,
  ylab = "Word probability",
  title = NULL,
  colors = NULL,
  measure_label = "Beta",
  ...
)
```

## Arguments

- top_topic_terms:

  A data frame containing topic terms with columns: topic, term, and
  beta. Typically created using get_topic_terms() or similar functions.

- topic_label:

  Optional topic labels. Can be either a named vector mapping topic
  numbers to labels, or a character string specifying a column name in
  top_topic_terms (default: NULL).

- ncol:

  Number of columns for facet wrap layout (default: 3).

- height:

  The height of the resulting Plotly plot, in pixels (default: 1200).

- width:

  The width of the resulting Plotly plot, in pixels (default: 800).

- ylab:

  Y-axis label (default: "Word probability").

- title:

  Plot title (default: NULL for auto-generated title).

- colors:

  Color palette for topics (default: NULL for auto-generated colors).

- measure_label:

  Label for the probability measure (default: "Beta").

- ...:

  Additional arguments passed to plotly::ggplotly().

## Value

A plotly object showing word probabilities faceted by topic.

## Examples

``` r
if (interactive()) {
  top_terms <- data.frame(
    topic = rep(1:2, each = 5),
    term = c("learning", "student", "education", "school", "teacher",
             "technology", "computer", "digital", "software", "system"),
    beta = c(0.05, 0.04, 0.03, 0.02, 0.01, 0.06, 0.05, 0.04, 0.03, 0.02)
  )
  plot <- plot_word_probability(top_terms)
  print(plot)
}
```
