# Create Label Selection UI Data

Creates a structured list for rendering label selection UI in Shiny.

## Usage

``` r
create_label_selection_data(label_candidates)
```

## Arguments

- label_candidates:

  List from generate_topic_labels_langgraph()

## Value

List of topic objects, each with:

- topic_number: Integer

- top_terms: Character vector

- candidates: List of candidate objects

## Examples

``` r
if (FALSE) { # \dontrun{
result <- generate_topic_labels_langgraph(...)
ui_data <- create_label_selection_data(result$label_candidates)
} # }
```
