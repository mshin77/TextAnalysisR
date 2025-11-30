# Format Label Candidates for Display

Helper function to format LangGraph label candidates for display in
Shiny UI.

## Usage

``` r
format_label_candidates(label_candidates)
```

## Arguments

- label_candidates:

  List of label candidate objects from generate_topic_labels_langgraph()

## Value

Data frame with columns:

- topic_index: Integer, topic number

- top_terms: Character, comma-separated top terms

- label: Character, suggested label

- reasoning: Character, LLM explanation

- candidate_number: Integer, candidate rank (1-3)

## Examples

``` r
if (FALSE) { # \dontrun{
result <- generate_topic_labels_langgraph(...)
df <- format_label_candidates(result$label_candidates)
print(df)
} # }
```
