# Calculate Text Readability

Calculates multiple readability metrics for texts including Flesch
Reading Ease, Flesch-Kincaid Grade Level, Gunning FOG index, and others.
Optionally includes lexical diversity metrics and sentence statistics.

## Usage

``` r
calculate_text_readability(
  texts,
  metrics = c("flesch", "flesch_kincaid", "gunning_fog"),
  include_lexical_diversity = TRUE,
  include_sentence_stats = TRUE,
  dfm_for_lexdiv = NULL,
  doc_names = NULL
)
```

## Arguments

- texts:

  Character vector of texts to analyze

- metrics:

  Character vector of readability metrics to calculate. Options:
  "flesch", "flesch_kincaid", "gunning_fog", "smog", "ari",
  "coleman_liau"

- include_lexical_diversity:

  Logical, include TTR and MTLD (default: TRUE)

- include_sentence_stats:

  Logical, include average sentence length (default: TRUE)

- dfm_for_lexdiv:

  Optional pre-computed DFM for lexical diversity calculation

- doc_names:

  Optional character vector of document names

## Value

A data frame with document names and readability scores

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c(
  "This is simple text.",
  "This sentence contains more complex vocabulary and structure."
)
readability <- calculate_text_readability(texts)
print(readability)
} # }
```
